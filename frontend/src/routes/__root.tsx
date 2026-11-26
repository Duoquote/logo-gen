import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router"
import type { QueryClient } from "@tanstack/react-query"
import { MoonIcon, SunIcon } from "lucide-react"

import { JobsMenu } from "@/components/jobs-menu"
import { useTheme } from "@/components/theme-provider"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { useLocalModels, useSettings } from "@/lib/queries"

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootLayout,
})

const NAV = [
  { to: "/", label: "Generate" },
  { to: "/designer", label: "Designer" },
  { to: "/gallery", label: "Gallery" },
  { to: "/models", label: "Models" },
  { to: "/settings", label: "Settings" },
] as const

function RootLayout() {
  const { theme, setTheme } = useTheme()
  const dark =
    theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)

  return (
    <div className="flex h-svh flex-col">
      <header className="flex h-12 shrink-0 items-center gap-6 border-b px-4">
        <Link to="/" className="flex items-center gap-2 font-heading text-sm font-semibold">
          <img src="/favicon.png" alt="" className="size-6" />
          logo-gen
        </Link>
        <nav className="flex items-center gap-1">
          {NAV.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              activeOptions={{ exact: n.to === "/" }}
              className="px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground data-[status=active]:text-foreground data-[status=active]:underline data-[status=active]:decoration-primary data-[status=active]:decoration-2 data-[status=active]:underline-offset-[18px]"
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <JobsMenu />
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Toggle dark mode"
            onClick={() => setTheme(dark ? "light" : "dark")}
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </Button>
        </div>
      </header>
      <SetupBanner />
      <main className="min-h-0 flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

function SetupBanner() {
  const settings = useSettings()
  const local = useLocalModels()
  if (!settings.data || !local.data) return null
  const hasKey = settings.data.secrets.OPENROUTER_KEY.set
  const hasLocal = local.data.models.some((m) => m.installed)
  if (hasKey || hasLocal) return null
  return (
    <div className="border-b p-3">
      <Alert>
        <AlertTitle>Nothing to generate with yet</AlertTitle>
        <AlertDescription>
          Add an OpenRouter API key in{" "}
          <Link to="/settings" className="underline">
            Settings
          </Link>{" "}
          for cloud models, or install a local model on the{" "}
          <Link to="/models" className="underline">
            Models
          </Link>{" "}
          page.
        </AlertDescription>
      </Alert>
    </div>
  )
}
