const benchmarks = require("./benchmarks");

module.exports = function () {
  const data = benchmarks();
  const map = {};
  for (const b of data) {
    const cat = b.category || "General";
    if (!map[cat]) map[cat] = { name: cat, slug: cat.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""), items: [] };
    map[cat].items.push(b);
  }
  return Object.values(map).sort((a, b) => a.name.localeCompare(b.name));
};
