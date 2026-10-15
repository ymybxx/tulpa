interface MainFrameOwner { mainFrame: unknown }

/** A local microphone action belongs only to the trusted top-level settings UI. */
export function assertOwnerVoiceSender(
  event: { sender: unknown; senderFrame: unknown },
  mainContents: MainFrameOwner | undefined,
): void {
  if (!mainContents || event.sender !== mainContents || event.senderFrame !== mainContents.mainFrame) {
    throw new Error("主人声纹操作仅允许从主窗口设置发起");
  }
}
