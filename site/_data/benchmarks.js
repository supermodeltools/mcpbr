const fs = require("fs");
const path = require("path");
const matter = require("gray-matter");

module.exports = function () {
  const dir = path.join(__dirname, "../../docs/benchmarks");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md"));
  return files.map((file) => {
    const raw = fs.readFileSync(path.join(dir, file), "utf-8");
    const { data, content } = matter(raw);
    const slug = file.replace(/\.md$/, "");
    return { ...data, slug, body: content };
  });
};
