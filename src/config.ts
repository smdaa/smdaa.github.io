import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://smdaa.github.io/",
  author: "Saad MDAA",
  profile: "https://github.com/smdaa/",
  desc: "My personal blog.",
  title: "",
  ogImage: "",
  lightAndDarkMode: false,
  postPerIndex: 8,
  postPerPage: 6,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: false,
  editPost: {
    url: "https://github.com/satnaing/astro-paper/edit/main/src/content/blog",
    text: "Suggest Changes",
    appendFilePath: false,
  },
};

export const LOCALE = {
  lang: "en", 
  langTag: ["en-EN"],
} as const;

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/smdaa/",
    linkTitle: `Github`,
    active: true,
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/saad-mdaa/",
    linkTitle: `LinkedIn`,
    active: true,
  },
];
