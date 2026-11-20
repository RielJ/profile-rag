/**
 * A JS-backed stream of `text`, in pieces of at most `size` characters.
 *
 * Tests use this rather than `new Response(text).body`: local workerd cannot
 * pipe a tee of an internal body into an IdentityTransformStream ("Inter-
 * TransformStream ReadableStream.pipeTo() is not implemented"). Workers AI's
 * stream pipes fine when deployed (the plan's spike, route /mix).
 */
export function streamOf(text: string, size = text.length || 1): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const pieces: string[] = [];
  for (let start = 0; start < text.length; start += size)
    pieces.push(text.slice(start, start + size));
  return new ReadableStream({
    start(controller) {
      for (const piece of pieces) controller.enqueue(encoder.encode(piece));
      controller.close();
    },
  });
}
