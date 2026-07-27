+++
title = "Building an RTL-SDR FM radio from scratch in C, part 1: the USB driver"
date = 2026-07-26
tags = ["c", "usb", "rtl-sdr"]
+++

{{< toc >}}

## Introduction

## The hardware

The device we will be talking to is a [Nooelec NESDR SMArt v5](https://www.nooelec.com/store/nesdr-smart-sdr.html), a USB dongle that covers radio frequencies from $100$ kHz to $1.75$ GHz with up to $3.2$ MHz of instantaneous bandwidth.

Inside it there are two chips:

![](/assets/building-an-rtl-sdr-fm-radio-from-scratch-in-c-part-1-the-usb-driver/simplified-schematic.png)

Block diagram from the [NESDR SMArt v5 datasheet](https://www.nooelec.com/store/downloads/dl/file/id/111/product/249/nesdr_smart_rtl_sdr_v5_datasheet_revision_1.pdf)

The first is the **R820T2** tuner. It is the analog front end: it takes whatever the antenna picks up across that entire $1.75$ GHz range and mixes the slice we care about down to a fixed intermediate frequency.

The second is the **RTL2832U**, a demodulator and USB interface. It digitizes the intermediate frequency coming out of the tuner and streams the resulting samples to the host over USB.

> **Fun fact:** the RTL2832U was built to receive DVB-T television. [Eric Fry found](https://osmocom.org/projects/rtl-sdr/wiki/rtl-sdr) that it can also be told to skip the TV decoding and hand over the raw I/Q samples instead, which is how a TV receiver became the cheapest SDR around.

Neither chip has a public datasheet. Everything we know about their registers comes from community reverse engineering, so whenever we need a register address or an init sequence we will go read [librtlsdr](https://github.com/osmocom/rtl-sdr) and treat it as the datasheet.

## Finding the USB device file

In Linux everything is a file, USB devices included.

Running `lsusb` shows us the dongle is on bus 3 as device 10:

```
$ lsusb
Bus 003 Device 010: ID 0bda:2838 Realtek Semiconductor Corp. RTL2838 DVB-T
```

So we can reach it through the file `/dev/bus/usb/003/010`

The catch is that this path is not stable. Device numbers are handed out at plug time, so they move around.

So we need an identifier that belongs to the device itself instead of one the kernel hands out. `lsusb` already printed it: `0bda:2838`, the vendor id and the product id.

Those two are burned into the device. `0bda` is Realtek, assigned to them by the USB-IF, and `2838` is the model. Unplug the dongle, move it to another port, reboot the machine, and it still answers `0bda:2838`.

To match on them we need the kernel's view of the device, which lives in `/sys/bus/usb/devices/`. Every connected device gets a folder there, and each property inside is a small text file:

```
$ ls /sys/bus/usb/devices/3-2/
busnum  devnum  idProduct  idVendor  manufacturer  product  serial  speed  ...

$ cat /sys/bus/usb/devices/3-2/idVendor
0bda
```

Which makes finding the dongle a one liner:

```
$ grep -l 2838 /sys/bus/usb/devices/*/idProduct
/sys/bus/usb/devices/3-2/idProduct
```

> **In C:** walk `/sys/bus/usb/devices/`, read `idVendor` and `idProduct` out of each folder, and stop at the one matching `0bda:2838`. That same folder holds `busnum` and `devnum`, which is what we need to build the `/dev/bus/usb/...` path.

## The kernel got there first

Okay so we have located the file, but it does not mean we get to use it yet.

As far as the kernel is concerned this is still a television receiver. So when we plugged it in, it went looking for a driver, found `dvb_usb_rtl28xxu`, and handed the device over to the driver:

```
$ basename $(readlink /sys/bus/usb/devices/3-2:1.0/driver)
dvb_usb_rtl28xxu
```

That driver is going to decode television for us, which is not what we want. We want the raw samples, so we need the interface for ourselves.

The usual lazy advice is to blacklist `dvb_usb_rtl28xxu` so it never loads. 

We will not, since the kernel lets us ask for the interface at runtime: we take it when we start and give it back when we are done.

The asking is done with the [`ioctl()`](https://man7.org/linux/man-pages/man2/ioctl.2.html) system call on the file we opened.

Kicking the driver off is `USBDEVFS_DISCONNECT` wrapped in a `struct usbdevfs_ioctl` together with the interface number.

```c
struct usbdevfs_ioctl cmd = {
    .ifno = 0,
    .ioctl_code = USBDEVFS_DISCONNECT,
    .data = NULL,
};
ioctl(fd, USBDEVFS_IOCTL, &cmd);
```

Claiming it afterwards is simpler, just the interface number:

```c
unsigned int ifno = 0;
ioctl(fd, USBDEVFS_CLAIMINTERFACE, &ifno);
```

On the way out we undo both, in reverse. First drop the claim with `USBDEVFS_RELEASEINTERFACE`, then, if we actually detached a driver earlier, give it back with the same wrapped call as before and `USBDEVFS_CONNECT` in place of `USBDEVFS_DISCONNECT`. 

> **In C:** open the device file, kick the driver off interface 0 with `USBDEVFS_DISCONNECT`, then claim it with `USBDEVFS_CLAIMINTERFACE`. On the way out, `USBDEVFS_RELEASEINTERFACE` and then `USBDEVFS_CONNECT` to hand it back.


