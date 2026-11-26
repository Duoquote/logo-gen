import { createFileRoute } from "@tanstack/react-router"

import { LocalModels } from "@/components/local-models"
import { ChatModelsTable, ImageModelsTable } from "@/components/openrouter-models"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useCatalog, useLocalModels, useSettings } from "@/lib/queries"

type Tab = "image" | "chat" | "local"

export const Route = createFileRoute("/models")({
  validateSearch: (s: Record<string, unknown>): { tab?: Tab } => ({
    tab: (["image", "chat", "local"].includes(s.tab as string) ? s.tab : undefined) as Tab | undefined,
  }),
  component: ModelsPage,
})

function ModelsPage() {
  const { tab = "image" } = Route.useSearch()
  const navigate = Route.useNavigate()
  const settings = useSettings()
  const catalog = useCatalog(settings.data?.secrets.OPENROUTER_KEY.set ?? false)
  const local = useLocalModels()

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <Tabs value={tab} onValueChange={(v) => navigate({ search: { tab: v as Tab } })}>
        <TabsList>
          <TabsTrigger value="image">Image models{catalog.data ? ` (${catalog.data.image.length})` : ""}</TabsTrigger>
          <TabsTrigger value="chat">Chat models{catalog.data ? ` (${catalog.data.chat.length})` : ""}</TabsTrigger>
          <TabsTrigger value="local">
            Local models{local.data ? ` (${local.data.models.filter((m) => m.installed).length} installed)` : ""}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="image" className="pt-3">
          <p className="mb-3 text-xs text-muted-foreground">
            OpenRouter models that output images, listed live with your key. Prices are per 1024 px image; models priced
            per token show an estimate.
          </p>
          <ImageModelsTable />
        </TabsContent>
        <TabsContent value="chat" className="pt-3">
          <p className="mb-3 text-xs text-muted-foreground">
            Text models for the designer chat and prompt writing. Prices per million tokens.
          </p>
          <ChatModelsTable />
        </TabsContent>
        <TabsContent value="local" className="pt-3">
          <LocalModels />
        </TabsContent>
      </Tabs>
    </div>
  )
}
