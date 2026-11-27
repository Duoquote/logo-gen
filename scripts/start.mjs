// `npm start` or `bun run start`: run the launcher for this system and pass the arguments on.
// Example: npm start -- --port 8000
import { spawnSync } from "node:child_process"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const root = join(dirname(fileURLToPath(import.meta.url)), "..")
const args = process.argv.slice(2)
const r =
  process.platform === "win32"
    ? spawnSync("cmd.exe", ["/d", "/c", join(root, "launch.cmd"), ...args], { stdio: "inherit", cwd: root })
    : spawnSync("sh", [join(root, "launch.sh"), ...args], { stdio: "inherit", cwd: root })
process.exit(r.status ?? 1)
