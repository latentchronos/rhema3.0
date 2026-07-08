import { useState } from "react"
import { invoke } from "@tauri-apps/api/core"
import { NotebookPenIcon } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

/**
 * Pre-service sermon notes (Bullet 5.2 / Gap F1). Verses the pastor mentions in
 * the notes get a +0.25 relevance boost in proactive suggestions. Without this
 * UI the priming index was always empty, so the boost never applied.
 */
export function SermonNotesDialog() {
  const [open, setOpen] = useState(false)
  const [notes, setNotes] = useState("")
  const [count, setCount] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const n = await invoke<number>("set_sermon_notes", { notes })
      setCount(n)
    } catch {
      setCount(null)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" title="Sermon notes (priming)">
          <NotebookPenIcon className="size-3.5" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Sermon notes</DialogTitle>
        <DialogDescription>
          Paste the pastor&apos;s notes before the service. Any verses referenced
          here are boosted in proactive suggestions.
        </DialogDescription>
        <Textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={10}
          placeholder="e.g. Romans 8:1-4, John 3:16, Galatians 5:22 — the renewing of the mind..."
          className="min-h-40 font-mono text-xs"
        />
        <div className="mt-2 flex items-center justify-end gap-3">
          {count !== null && (
            <span className="text-xs text-muted-foreground">
              {count} verse{count === 1 ? "" : "s"} primed
            </span>
          )}
          <Button onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save notes"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
