+++
title = "Building an RTL-SDR FM radio from scratch in C, part 2: bringing up the RTL2832U"
date = 2026-09-01
draft = true
tags = ["c", "rtl-sdr", "reverse-engineering"]
+++

{{< toc >}}

## Introduction

This is the second part of a series about building an FM radio from scratch in C, without using any third-party library.

The hardware is an RTL-SDR dongle, and it has two halves. The R820T2 tuner picks up the radio signal and brings it down in frequency. The RTL2832U digitizes it, filters it, and sends it to the PC over USB.

[Part 1]({{< relref "building-an-rtl-sdr-fm-radio-from-scratch-in-c-part-1-the-usb-driver.md" >}}) covered talking to the RTL2832U over USB: finding the dongle, taking it from the kernel driver, and reading and writing the chip's registers. This part sets up the RTL2832U until it sends raw I/Q samples. The tuner comes in part 3.

We follow the writes [librtlsdr](https://github.com/osmocom/rtl-sdr) makes, and use the comments in the [Linux kernel driver](https://github.com/torvalds/linux/tree/master/drivers/media) to understand them.

You can find the full source code [here](https://github.com/smdaa/sdr).

## Register access

A register is addressed by three things: a **block**, a **page**, and an **address**.

Each block is a separate part of the chip. librtlsdr [numbers them](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L395-L403) from `0` to `6`:

| block | name        | function                                     |
| ----- | ----------- | -------------------------------------------- |
| `0`   | Demodulator | signal processing: mode, filter, sample rate |
| `1`   | USB         | the endpoint that sends samples to the host  |
| `2`   | System      | power, clock and reset of the demodulator    |
| `3`   | Tuner       | not used by librtlsdr or the kernel          |
| `4`   | ROM         | not used by librtlsdr or the kernel          |
| `5`   | IR          | infrared receiver for a remote control       |
| `6`   | I2C         | the bus to the tuner chip                    |

The demodulator has more than 256 registers, but its register address is only one byte. So its registers are split into pages of 256.

As a reminder, `wValue` and `wIndex` hold the parameters of the USB control transfer. `wIndex` contains the block and the page, and `wValue` contains the address:

```c
struct usbdevfs_ctrltransfer ctrl = {
    .bRequestType = 0xc0,
    .bRequest     = 0,
    .wValue       = 0x3000, // DEMOD_CTL
    .wIndex       = 0x0200, // read from the system block
    .wLength      = 1,
    .timeout      = 1000,
    .data         = &value};
```

`wIndex` is 16 bits, which is four hex digits. Each digit is a separate field:

| digit | field  | meaning                      |
| ----- | ------ | ---------------------------- |
| 3     |        | unused                       |
| 2     | block  | which part of the chip       |
| 1     | access | `1` to write, `0` to read    |
| 0     | page   | which page, demodulator only |

So the `0x0200` above reads from block `2`, the system block, at page `0`, and `0x0210` writes to it.

Every block other than the demodulator leaves the page at `0`.

Therefore building `wIndex` is done with this function:

```c
static uint16_t reg_windex(uint8_t block, uint8_t page, int write)
{
    return (uint16_t)((block << 8) | (write ? 0x10 : 0x00) | (page & 0x0f));
}
```

`wValue` contains the address. For the demodulator, the address goes in the high byte and the low byte is always `0x20`, as librtlsdr does in [`rtlsdr_demod_read_reg`](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L518):

```c
static uint16_t reg_wvalue(uint8_t block, uint16_t addr)
{
    return (block == DEMODB) ? (uint16_t)((addr << 8) | 0x20) : addr;
}
```

`rtl_read_reg` and `rtl_write_reg` are the backbone of this article. They wrap the USB control transfer and use `reg_windex` and `reg_wvalue` to fill `wIndex` and `wValue`.

Every register holds [one byte](https://github.com/torvalds/linux/blob/master/drivers/media/dvb-frontends/rtl2832.c#L1060), so `wLength` is `1`. A value wider than one byte is split over registers at consecutive addresses.

To read, `bRequestType` is `0xc0`, a vendor request from the device (the dongle) to the host (the PC):

```c
int rtl_read_reg(rtl_dev *dev, uint8_t block, uint8_t page, uint16_t addr, uint8_t *out)
{
    struct usbdevfs_ctrltransfer ctrl = {
        .bRequestType = 0xc0,
        .bRequest = 0,
        .wValue = reg_wvalue(block, addr),
        .wIndex = reg_windex(block, page, 0),
        .wLength = 1,
        .timeout = 1000,
        .data = out};

    int ret = ioctl(dev->fd, USBDEVFS_CONTROL, &ctrl);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to read register 0x%04x of block %u page %u: %s\n",
                addr, block, page, strerror(errno));
    }
    return ret;
}
```

To write, `bRequestType` is `0x40`, a vendor request from the host (the PC) to the device (the dongle), and `data` holds the byte to write:

```c
int rtl_write_reg(rtl_dev *dev, uint8_t block, uint8_t page, uint16_t addr, uint8_t val)
{
    struct usbdevfs_ctrltransfer ctrl = {
        .bRequestType = 0x40,
        .bRequest = 0,
        .wValue = reg_wvalue(block, addr),
        .wIndex = reg_windex(block, page, 1),
        .wLength = 1,
        .timeout = 1000,
        .data = &val};

    int ret = ioctl(dev->fd, USBDEVFS_CONTROL, &ctrl);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to write register 0x%04x of block %u page %u: %s\n",
                addr, block, page, strerror(errno));
    }
    return ret;
}
```

Both return the number of bytes transferred, so `1` on success.

## Powering on the demodulator

librtlsdr [powers on the demodulator with two writes](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L622-L624) to the system block: `DEMOD_CTL_1 = 0x22` and `DEMOD_CTL = 0xe8`.

The kernel's USB driver, `rtl28xxu.c`, names the bits of `DEMOD_CTL` in its comments:

| bit  | kernel comment                      |
| ---- | ----------------------------------- |
| 7    | ["enable PLL"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1565-L1566)     |
| 6, 3 | ["enable ADC"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1632)           |
| 5    | ["disable reset"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1570-L1571)  |

`0xe8` is `1110 1000`: PLL on, ADC on, reset released.

No public source explains `DEMOD_CTL_1`, so we cannot decode `0x22`.

We then define our power-on function as:

```c
int rtl_power_on(rtl_dev *dev)
{
    if (rtl_write_reg(dev, SYSB, 0, DEMOD_CTL_1, 0x22) != 1 ||
        rtl_write_reg(dev, SYSB, 0, DEMOD_CTL, 0xe8) != 1)
    {
        return -1;
    }
    return 0;
}
```

## Switching to SDR mode

After powering on the demodulator, librtlsdr switches it from TV mode to SDR mode with a series of writes, [each with a comment](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L626-L665):

| page | register      | value           | librtlsdr comment                                                  | in plain words                                                                                                                        |
| ---- | ------------- | --------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `0x01`        | `14`, then `10` | "reset demod"                                                      |                                                                                                                                       |
| 1    | `0x15`        | `00`            | "disable spectrum inversion and adjacent channel rejection"        | Keep the spectrum as is, not mirrored, and stop filtering out neighboring TV channels.                                                |
| 1    | `0x16`–`0x1b` | `00`            | "clear both DDC shift and IF frequency registers"                  | Set the digital down-converter (DDC) shift and the intermediate frequency (IF) to 0 Hz.                                               |
| 0    | `0x19`        | `05`            | "enable SDR mode, disable DAGC"                                    | Send raw I/Q samples over USB instead of decoded TV data. Turn off digital automatic gain control (DAGC).                             |
| 1    | `0x93` `0x94` | `f0` `0f`       | "init FSM state-holding register"                                  | Set the starting value of an internal finite state machine (FSM). No source says what it controls.                                    |
| 1    | `0x11`        | `00`            | "disable AGC"                                                      | Turn off automatic gain control (AGC), which keeps the signal level constant.                                                         |
| 1    | `0x04`        | `00`            | "disable RF and IF AGC loop"                                       | Stop the demodulator from controlling the tuner's RF and IF gain.                                                                     |
| 0    | `0x61`        | `60`            | "disable PID filter"                                               | Turn off the packet ID (PID) filter, which passes only the packets of selected TV streams.                                            |
| 0    | `0x06`        | `80`            | "default ADC_I/ADC_Q datapath"                                     |                                                                                                                                       |
| 1    | `0xb1`        | `1b`            | "Enable Zero-IF mode, DC cancellation, IQ estimation/compensation" | Expect a zero-IF input, centered on 0 Hz. Remove the DC offset and correct the I/Q imbalance.                                         |
| 0    | `0x0d`        | `83`            | "disable 4.096 MHz clock output on pin TP_CK0"                     |                                                                                                                                       |

Our `rtl_set_sdr_mode` makes the same writes, in the same order:

```c
static const struct
{
    uint8_t page, addr, val;
} sdr_mode[] = {
    {1, 0x01, 0x14}, {1, 0x01, 0x10},
    {1, 0x15, 0x00},
    {1, 0x16, 0x00}, {1, 0x17, 0x00},
    {1, 0x18, 0x00}, {1, 0x19, 0x00},
    {1, 0x1a, 0x00}, {1, 0x1b, 0x00},
    {0, 0x19, 0x05},
    {1, 0x93, 0xf0}, {1, 0x94, 0x0f},
    {1, 0x11, 0x00},
    {1, 0x04, 0x00},
    {0, 0x61, 0x60},
    {0, 0x06, 0x80},
    {1, 0xb1, 0x1b},
    {0, 0x0d, 0x83},
};

int rtl_set_sdr_mode(rtl_dev *dev)
{
    for (size_t i = 0; i < sizeof(sdr_mode) / sizeof(sdr_mode[0]); i++)
    {
        if (rtl_write_reg(dev, DEMODB, sdr_mode[i].page, sdr_mode[i].addr, sdr_mode[i].val) != 1)
        {
            return -1;
        }
    }
    return 0;
}
```

## Loading the low-pass filter

librtlsdr also writes a filter to page 1, registers `0x1c` to `0x2f`. The filter works on the samples coming out of the ADC, before the chip lowers the sample rate.

It is a FIR filter: each output sample is the sum of the last [32 input samples](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L71-L83), each multiplied by a coefficient.

It is a low-pass filter: it passes about 1.2 MHz on each side of the center and cuts the rest (computed from the coefficients). That stops the rest from aliasing onto our band when the chip lowers the rate. We explain aliasing in the next part.

librtlsdr sends [16 coefficients](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L84-L87). The other 16 are the same values in reverse order:

```c
static const int16_t fir_default[16] = {
    -54, -36, -41, -40, -32, -14, 14, 53,
    101, 156, 215, 273, 327, 372, 404, 421};
```

The first 8 coefficients take one byte each. The last 8 need [12 bits](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L78-L79) each, so they are packed in pairs: two coefficients in three bytes. That makes 20 bytes, written to page 1 from `0x1c` to `0x2f`:

```c
int rtl_set_fir(rtl_dev *dev)
{
    uint8_t fir[20];

    for (uint16_t i = 0; i < 8; i++)
    {
        fir[i] = (uint8_t)fir_default[i];
    }

    /* Join each pair of 12-bit values, abc and def in hex, into abcdef, then
     * split it into the bytes ab cd ef. */
    for (int k = 0; k < 4; k++)
    {
        const uint32_t pair = (uint32_t)(fir_default[8 + 2 * k] & 0xfff) << 12 |
                              (uint32_t)(fir_default[9 + 2 * k] & 0xfff);

        fir[8 + 3 * k] = (uint8_t)(pair >> 16);
        fir[9 + 3 * k] = (uint8_t)(pair >> 8);
        fir[10 + 3 * k] = (uint8_t)pair;
    }

    for (uint16_t i = 0; i < 20; i++)
    {
        if (rtl_write_reg(dev, DEMODB, 1, 0x1c + i, fir[i]) != 1)
        {
            return -1;
        }
    }

    return 0;
}
```

## USB setup

librtlsdr starts its bring-up with [three writes to the USB block](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L617-L620). They set up endpoint `0x81`, where the samples come out.

The kernel's USB driver, `rtl28xxu.c`, names each write in its comments:

| register         | bytes   | kernel comment                                                                                                                         | in plain words                                                                                                                     |
| ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `USB_SYSCTL`     | `09`    | ["enable DMA and Full Packet Mode"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1464)       | Turn on DMA (direct memory access): a dedicated circuit copies the samples to the endpoint as they arrive. Send only full packets. |
| `USB_EPA_MAXPKT` | `00 02` | ["set EPA maximum packet size to 0x0200"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1470) | EPA (endpoint A) is endpoint `0x81`. Each packet carries up to 512 bytes.                                                          |
| `USB_EPA_CTL`    | `10 02` | ["streaming EP: set stall & reset"](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c#L1594)       | EP is endpoint. Pause it (stall) and empty its queue of waiting samples (reset).                                                   |

`0x0200` is 512, the packet size of endpoint `0x81` from part 1, stored low byte first: `00 02`.

Our `rtl_init_usb` makes the same writes:

```c
int rtl_init_usb(rtl_dev *dev)
{
    if (rtl_write_reg(dev, USBB, 0, USB_SYSCTL, 0x09) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_MAXPKT, 0x00) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_MAXPKT + 1, 0x02) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_CTL, 0x10) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_CTL + 1, 0x02) != 1)
    {
        return -1;
    }
    return 0;
}
```

`rtl_init` runs the whole bring-up in librtlsdr's order:

```c
int rtl_init(rtl_dev *dev)
{
    if (rtl_init_usb(dev) != 0 ||
        rtl_power_on(dev) != 0 ||
        rtl_set_sdr_mode(dev) != 0 ||
        rtl_set_fir(dev) != 0)
    {
        return -1;
    }

    printf("RTL2832U initialized, SDR mode on\n");
    return 0;
}
```

## Setting the sample rate

A resampler in the chip turns samples at the crystal frequency, 28.8 MHz, into samples at the rate we ask for. librtlsdr sets it with [the ratio between the two rates](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L1106-L1107), written to page 1 from `0x9f` to `0xa2`. Our `rtl_set_sample_rate` does the same:

```c
int rtl_set_sample_rate(rtl_dev *dev, uint32_t rate)
{
    if (rate <= 225000 || rate > 3200000 || (rate > 300000 && rate <= 900000))
    {
        fprintf(stderr, "Unsupported sample rate: %u Hz\n", rate);
        return -1;
    }

    /* crystal ticks per sample, with 22 fractional bits */
    const uint32_t ratio = (uint32_t)(((uint64_t)RTL_XTAL_HZ << 22) / rate) & 0x0ffffffc;

    if (rtl_write_reg(dev, DEMODB, 1, 0x9f, ratio >> 24) != 1 ||
        rtl_write_reg(dev, DEMODB, 1, 0xa0, (ratio >> 16) & 0xff) != 1 ||
        rtl_write_reg(dev, DEMODB, 1, 0xa1, (ratio >> 8) & 0xff) != 1 ||
        rtl_write_reg(dev, DEMODB, 1, 0xa2, ratio & 0xff) != 1 ||
        /* 0 ppm correction */
        rtl_write_reg(dev, DEMODB, 1, 0x3f, 0x00) != 1 ||
        rtl_write_reg(dev, DEMODB, 1, 0x3e, 0x00) != 1 ||
        /* soft reset */
        rtl_write_reg(dev, DEMODB, 1, 0x01, 0x14) != 1 ||
        rtl_write_reg(dev, DEMODB, 1, 0x01, 0x10) != 1)
    {
        return -1;
    }
    return 0;
}
```

## Starting the stream

librtlsdr starts the stream by [writing `USB_EPA_CTL` twice](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L1702-L1711): stall and reset, which empties the queue, then `00 00`, which unpauses the endpoint. Our `rtl_start_stream` does the same:

```c
int rtl_start_stream(rtl_dev *dev)
{
    if (rtl_write_reg(dev, USBB, 0, USB_EPA_CTL, 0x10) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_CTL + 1, 0x02) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_CTL, 0x00) != 1 ||
        rtl_write_reg(dev, USBB, 0, USB_EPA_CTL + 1, 0x00) != 1)
    {
        return -1;
    }
    return 0;
}
```

We read the samples from endpoint `0x81` with the `USBDEVFS_BULK` ioctl:

```c
int rtl_read(rtl_dev *dev, uint8_t *buf, unsigned int len)
{
    struct usbdevfs_bulktransfer bulk = {
        .ep = (unsigned int)dev->bulk_in_ep,
        .len = len,
        .timeout = 1000,
        .data = buf};

    int ret = ioctl(dev->fd, USBDEVFS_BULK, &bulk);
    if (ret < 0)
    {
        fprintf(stderr, "Failed to read samples: %s\n", strerror(errno));
        return -1;
    }
    return ret;
}
```

`save_samples` writes the stream to a file, as it comes:

```c
static int save_samples(rtl_dev *dev, const char *path, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL)
    {
        perror(path);
        return -1;
    }

    size_t done = 0;
    while (done < len)
    {
        int n = rtl_read(dev, buf, READ_LEN);
        if (n < 0)
        {
            fclose(f);
            return -1;
        }

        size_t keep = len - done < (size_t)n ? len - done : (size_t)n;
        if (fwrite(buf, 1, keep, f) != keep)
        {
            perror(path);
            fclose(f);
            return -1;
        }
        done += keep;
    }

    return fclose(f) == 0 ? 0 : -1;
}
```

`plot.py` reads the file back, and plots I and Q over time on top and the spectrum below:

```python
def load(path):
    raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
    return (raw[0::2] - 127.5) / 127.5 + 1j * (raw[1::2] - 127.5) / 127.5


def plot(iq, rate, out=None, n_time=2000, n_fft=4096):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7))

    t = np.arange(n_time) / rate * 1e6
    top.plot(t, iq[:n_time].real, label="I")
    top.plot(t, iq[:n_time].imag, label="Q")
    top.set_xlabel("time (µs)")
    top.set_ylabel("amplitude")
    top.set_ylim(-1, 1)
    top.legend()

    blocks = iq[: len(iq) // n_fft * n_fft].reshape(-1, n_fft)
    power = np.mean(np.abs(np.fft.fft(blocks * np.hanning(n_fft), axis=1)) ** 2, axis=0)
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / rate)) / 1e6
    bottom.plot(freq, 10 * np.log10(np.fft.fftshift(power) + 1e-20))
    bottom.set_xlabel("frequency from the center (MHz)")
    bottom.set_ylabel("power (dB)")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    else:
        plt.show()

```

One second at 2.048 MS/s:

![One second of samples at 2.048 MS/s. Top: I is a flat line at zero, Q is small noise around zero. Bottom: the spectrum, symmetric around the center, with a comb of lines near zero and two strong pairs of lines at about 0.64 and 0.92 MHz on each side](/assets/building-an-rtl-sdr-fm-radio-from-scratch-in-c-part-2-bringing-up-the-rtl2832u/samples.png)

Each sample is two bytes, I then Q. I is flat because the tuner isn't set up yet. Q is not: on this dongle, the antenna's 0.1 to 30 MHz goes [straight to the RTL2832U]({{< relref "building-an-rtl-sdr-fm-radio-from-scratch-in-c-part-1-the-usb-driver.md" >}}), and it lands on Q. Part 3 sets up the tuner.

## Where we are

The RTL2832U is up: powered on, in SDR mode, and streaming samples at the rate we ask for.

The samples carry no FM yet, because the tuner isn't set up. Part 3 explains the receiver from the antenna to the I/Q samples, and sets up the tuner.

<!--
librtlsdr starts its bring-up with [three writes to the USB block](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L617-L620). They set up the sample endpoint, `0x81`, and `rtl_init` copies them. The kernel's USB driver, [`rtl28xxu.c`](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c), writes the same three registers, with a comment on each:

| register         | value    | kernel comment                          |
| ---------------- | -------- | --------------------------------------- |
| `USB_SYSCTL`     | `0x09`   | "enable DMA and Full Packet Mode"       |
| `USB_EPA_MAXPKT` | `0x0002` | "set EPA maximum packet size to 0x0200" |
| `USB_EPA_CTL`    | `0x1002` | "streaming EP: set stall & reset"       |

We send two-byte values high byte first, as librtlsdr does, so the chip gets the same bytes. The packet size is 512, `0x0200`, but it is written as `0x0002` because the USB block takes the first byte as the low byte.

## Initial state

Before writing anything, we read the registers to see the state of the chip.

librtlsdr [powers the chip on through two registers](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c#L622-L624) in the system block, `DEMOD_CTL` and `DEMOD_CTL_1`. When we plug the dongle in, they read:

```
DEMOD_CTL = 0x20   DEMOD_CTL_1 = 0x02
```

`DEMOD_CTL` controls the demodulator. The kernel's USB driver for the dongle, [`rtl28xxu.c`](https://github.com/torvalds/linux/blob/master/drivers/media/usb/dvb-usb-v2/rtl28xxu.c), names its bits in comments: it sets `0x80` to "enable PLL", `0x20` to "disable reset", and `0x48` to "enable ADC", one bit for each of the two ADCs:

| bit  | meaning                     |
| ---- | --------------------------- |
| 7    | demodulator PLL enable      |
| 6, 3 | ADC enable, one bit per ADC |
| 5    | reset, `1` releases it      |

`0x20` is `0010 0000`: PLL off, both ADCs off, reset released.

No public source explains `DEMOD_CTL_1`. The kernel's only comment on it is "suspend?", so `0x02` cannot be decoded.

The value also depends on what ran before. The kernel's own driver for the dongle, `dvb_usb_rtl28xxu`, configures the chip when the dongle is plugged in, and again when our program releases it. So `0x20` is the value right after a replug. Run a program that powers the chip on, run it again without unplugging, and the second run reads `0x68`. The kernel's power-off code clears only the PLL bit, so the ADCs are still on. Every cold reading below starts from `0x20`.

## Change one write at a time

When a working driver performs several writes and it is not clear which one matters, perform them one at a time and read back after each.

The demodulator digitizes the signal from the tuner and decodes it. Two ADCs, analog-to-digital converters, sample the input, one for I and one for Q. On a television receiver the decoding is DVB-T.

The demodulator is block `0`, and reading it fails. Page 1, read register by register, with `--` for each failed read:

```
demodulator page 1, 0x00 to 0x1f:
  00: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
```

Each read returns `-1` with `errno` set to `EPIPE`. On a control transfer this means the device stalled the endpoint, which is how a USB device rejects a request it cannot handle. No value comes back.

A stall and a wrong value mean different things. A wrong value would point to a wrong address. A stall means the target does not respond, and the system block responds on the same endpoint, so the addressing is correct. The demodulator is not on the bus.

librtlsdr powers it on with two writes to the system block, in this order:

```c
rtl_write_reg(dev, BLOCK_SYS, 0, DEMOD_CTL_1, 0x22, 1);
rtl_write_reg(dev, BLOCK_SYS, 0, DEMOD_CTL,   0xe8, 1);
```

`0xe8` sets all four bits above: the PLL, both ADCs and the reset release. `0x22` cannot be decoded, and librtlsdr writes it without a comment.

We test the two writes separately, reading the demodulator after each. First `DEMOD_CTL_1`, from a cold chip:

```
cold:                DEMOD_CTL = 0x20   DEMOD_CTL_1 = 0x02
cold:                demodulator page 0 STALLS, page 1 STALLS

writing DEMOD_CTL_1 = 0x22
after DEMOD_CTL_1:   DEMOD_CTL = 0x20   DEMOD_CTL_1 = 0x02
after DEMOD_CTL_1:   demodulator page 0 STALLS, page 1 STALLS
```

The demodulator still stalls, and `DEMOD_CTL_1` still reads `0x02`: the write did not take effect. Only the read-back shows this.

Then `DEMOD_CTL`, from a cold chip:

```
cold:                DEMOD_CTL = 0x20   DEMOD_CTL_1 = 0x02
cold:                demodulator page 0 STALLS, page 1 STALLS

writing DEMOD_CTL = 0xe8
after DEMOD_CTL:     DEMOD_CTL = 0xe8   DEMOD_CTL_1 = 0x02
after DEMOD_CTL:     demodulator page 0 answers (0x19 = 20), page 1 answers (0x01 = 10)
```

After this write the demodulator responds. Starting from `0x20`, the reset was already released, so the only bits added are the PLL and the two ADCs. The ADCs feed the demodulator and do not control access to it. That leaves the PLL. A PLL generates a clock, and this one clocks the demodulator. Every register access is handled by logic inside the block, and that logic only runs while it has a clock. Without one, the block stalls on reads and writes alike.

The write that puts the demodulator on the bus is the one that starts its clock. None of the three sources state this.

Page 1 then reads:

```
demodulator page 1, 0x00 to 0x1f:
  00: 01 10 40 2d c0 be 10 0f 80 7f 80 7f 00 00 00 00
  10: 0a 07 39 10 04 06 3f ce cc 35 d7 5d 09 f6 d2 a7
```

## Compare two drivers

Besides librtlsdr, the kernel has an SDR driver for the RTL2832U, [`rtl2832_sdr.c`](https://github.com/torvalds/linux/blob/master/drivers/media/dvb-frontends/rtl2832_sdr.c).

After power-on the demodulator is still configured for DVB-T. librtlsdr reconfigures it in `rtlsdr_init_baseband`, and `rtl_init` performs the same writes. Four of those registers, read after power-on and after `rtl_init`:

| register                  | after power-on | after init | what it is    |
| ------------------------- | -------------- | ---------- | ------------- |
| page 0, `0x19`            | `20`           | `05`       | mode          |
| page 0, `0x61`            | `e0`           | `60`       | PID filter    |
| page 1, `0x93` and `0x94` | `0f ff`        | `f0 0f`    | FSM           |
| page 1, `0x04`            | `c0`           | `00`       | RF and IF AGC |

The kernel driver shows which column is which. When it starts, it writes the right-hand values to the first three registers. When it stops, it writes the left-hand values, `20`, `e0` and `0f ff`, back, to return the chip to TV mode. So the left column is TV mode, and both drivers use the same values for SDR mode.

librtlsdr's comment on writing `05` to the mode register is "enable SDR mode".

DVB-T sends television as packets tagged with a packet ID, and the PID filter drops packets with unwanted IDs. The kernel uses bit 7 of `0x61` as the filter enable, and `e0` to `60` clears it.

Neither driver explains the FSM registers. FSM stands for finite state machine; librtlsdr calls them an "FSM state-holding register" and the kernel names them `SM_PASS`. Both write the same values. That agreement is the only evidence for them, since neither driver explains them.

The drivers disagree on the AGC register. The AGC loops adjust gain automatically to hold a TV signal at a constant level. The kernel's register table names bits 7 and 6 of `0x04` as the IF and RF AGC enables. librtlsdr writes `00`, disabling both. The kernel SDR driver writes `cc` for this tuner, leaving both enabled. The code follows librtlsdr. Which setting works better can only be tested with a signal, after part 3.

`rtl_init` writes further registers, listed in `rtl.c`. The IF frequency, spectrum inversion and zero-IF among them depend on the tuner and are set in part 3.

Comparing also exposes comments that do not match the bits. librtlsdr writes `80` to page 0 `0x06` with the comment "opt_adc_iq = 0", but the register already read `00`, so `opt_adc_iq` was already zero. The bit the write actually sets is bit 7, which the kernel names `MPEG_IO_OPT_2_2`. Only reading the register before and after the write shows this.

For the filter, librtlsdr computes the register bytes from a list of coefficients, and the kernel SDR driver writes them as a constant string. Computing the bytes independently and matching both verifies the packing code.

It is a low-pass filter: it passes low frequencies and removes high ones. It works by outputting a weighted sum of the last 32 input samples, and the coefficients are the weights. librtlsdr's coefficients, described in its source as "used for DAB/FM by the Windows driver, the DVB driver uses different ones":

```c
static const int16_t fir_default[16] = {
    -54, -36, -41, -40, -32, -14, 14, 53, // 8 bit signed
    101, 156, 215, 273, 327, 372, 404, 421};
```

The 32 coefficients are symmetric: the second 16 are the first 16 in reverse order. So only the first 16 are sent, and the chip mirrors them.

The first 8 are 8-bit signed, one byte each. Negative values are stored in two's complement, so -54 is stored as 256 - 54 = 202, which is `ca`. 53 is `35`.

The next 8 are 12-bit signed, three hex digits each, so two fit in three bytes. In the first pair, 101 is `065` and 156 is `09c`. Concatenated they give `06509c`, which splits into the bytes `06 50 9c`.

`rtl_set_fir` does this with shifts, `v0` and `v1` being the pair:

```c
out[0] = (uint8_t)(v0 >> 4);
out[1] = (uint8_t)((v0 << 4) | ((v1 >> 8) & 0x0f));
out[2] = (uint8_t)v1;
```

Truncated to a byte, `v0 >> 4` is `06`, `(v0 << 4) | (v1 >> 8)` is `50` and `v1` is `9c`.

This gives 20 bytes, written to page 1 from `0x1c` to `0x2f`. Read back after `rtl_set_fir`:

```
ca dc d7 d8 e0 f2 0e 35 06 50 9c 0d 71 11 14 71 74 19 41 a5
```

The kernel SDR driver writes the same 20 bytes, as the string `"\xca\xdc\xd7\xd8..."`.

## Measure the result

The final check is a measurable quantity: the rate at which samples arrive. It is only correct if both the sample rate setting and the USB configuration work.

The filter runs at the crystal frequency, 28.8 MHz. A resampler turns that stream into one at the requested rate, and it is set by the ratio between the two rates. librtlsdr computes the ratio with 22 bits after the binary point, and writes it to page 1 starting at `0x9f`:

```c
const uint32_t ratio = (uint32_t)(((uint64_t)RTL_XTAL_HZ << 22) / rate) & 0x0ffffffc;
```

For 2.048 MHz the ratio is $28.8 / 2.048 = 14.0625$, which with 22 fractional bits is $14.0625 \times 2^{22}$ = `0x03840000`.

`rtl_set_sample_rate` writes the ratio, zeroes the sample rate correction in `0x3e` and `0x3f`, and soft resets the demodulator, as librtlsdr does.

`rtl_reset_buffer` then takes the endpoint out of stall by writing `USB_EPA_CTL` twice: stalled with the FIFO reset, which empties the FIFO, then zero:

```c
rtl_write_reg(dev, BLOCK_USB, 0, USB_EPA_CTL, USB_EPA_CTL_STALL_RESET, 2);
rtl_write_reg(dev, BLOCK_USB, 0, USB_EPA_CTL, USB_EPA_CTL_RUN, 2);
```

Samples are read with a bulk transfer, using the `USBDEVFS_BULK` ioctl on the endpoint from part 1:

```c
struct usbdevfs_bulktransfer bulk = {
    .ep      = (unsigned int)dev->bulk_in_ep,
    .len     = len,
    .timeout = 1000,
    .data    = buf};

int ret = ioctl(dev->fd, USBDEVFS_BULK, &bulk);
```

The first read at 2.048 MHz:

```
read 262144 bytes, the first 16: 7f 80 7f 80 7f 80 7f 80 7f 80 7f 80 7f 80 7f 80
```

Each sample is one byte of I followed by one byte of Q. I and Q are two versions of the signal, taken against references a quarter of a cycle apart, and together they give both its strength and its phase. Each byte is unsigned, with zero at the midpoint of 0 to 255. These values, `7f` and `80`, sit at the midpoint because nothing is tuned.

Timing the reads verifies the stream. Each sample is two bytes, so samples per second is bytes per second divided by two:

| asked for | measured  |
| --------- | --------- |
| 1,024,000 | 1,024,395 |
| 2,048,000 | 2,048,601 |
| 2,400,000 | 2,400,636 |

The measured rates match the requested rates to within 0.04%.

## Where we are

The code can address any register on the chip, power on the demodulator, configure it out of TV mode, load the filter, and read samples at a requested rate.

It cannot tune yet, so the samples contain no signal. The tuner is a separate chip, not on the USB bus. It is on an [I2C](https://en.wikipedia.org/wiki/I%C2%B2C) bus inside the dongle, a two-wire bus that chips use to talk to each other.

Both drivers reach the tuner through an I2C repeater in the RTL2832U, which they turn on before each tuner access and off after. In librtlsdr this is `rtlsdr_set_i2c_repeater`. The kernel calls it the "I2C gate/mux/repeater" and names the bit `DVBT_IIC_REPEAT`: bit 3 of page 1, `0x01`. After power-on that register reads `10`, so the repeater is off.

Part 3 enables the repeater, locates the tuner, and sets the IF frequency, spectrum inversion and zero-IF for it.
-->
