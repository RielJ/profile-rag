// openapi-typescript 7 builds its output with the TypeScript compiler API.
// TypeScript 7 (this project's compiler) has no JavaScript API, and
// openapi-typescript's peer range is ^5.x. This hook gives openapi-typescript
// its own TypeScript 5 instead of the project's, which stays on 7.
module.exports = {
  hooks: {
    readPackage(pkg) {
      if (pkg.name === "openapi-typescript") {
        delete pkg.peerDependencies?.typescript;
        pkg.dependencies = { ...pkg.dependencies, typescript: "5.9.3" };
      }
      return pkg;
    },
  },
};
