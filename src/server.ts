import { NanoBananaMCP } from "./index.js";

const server = new NanoBananaMCP();
server.run().catch(console.error);
