import { afterEach, describe, expect, it, vi } from "vitest"

const invokeMock = vi.fn()

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}))

describe("acquireOperatorLock", () => {
  afterEach(() => {
    invokeMock.mockReset()
  })

  it("invokes the acquire_operator_lock command", async () => {
    invokeMock.mockResolvedValue(1)
    const { acquireOperatorLock } = await import("./operator-lock")
    acquireOperatorLock()
    expect(invokeMock).toHaveBeenCalledWith("acquire_operator_lock")
  })

  it("swallows backend errors (best-effort)", async () => {
    invokeMock.mockRejectedValue(new Error("backend down"))
    const { acquireOperatorLock } = await import("./operator-lock")
    expect(() => acquireOperatorLock()).not.toThrow()
  })
})
