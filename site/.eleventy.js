const markdownIt = require("markdown-it");

module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy("static");

  const md = markdownIt({ html: true, linkify: true });
  eleventyConfig.addFilter("markdown", (content) => content ? md.render(content) : "");
  eleventyConfig.addFilter("slugify", (str) => str ? str.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") : "");
  eleventyConfig.addFilter("lower", (str) => str ? str.toLowerCase() : "");
  eleventyConfig.addFilter("json", (value) => JSON.stringify(value));

  return {
    dir: { input: ".", includes: "_includes", data: "_data", output: "_site" },
  };
};
