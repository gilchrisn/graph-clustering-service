// // src/test/setup.ts - Clean fixes for remaining issues
// import '@testing-library/jest-dom';
// import { vi, expect } from 'vitest';

// // Type declarations for global objects
// declare global {
//   var fetch: any;
//   var ResizeObserver: any;
//   var IntersectionObserver: any;
//   var File: any;
//   var FileReader: any;
//   var FormData: any;
//   var WebSocket: any;
//   namespace NodeJS {
//     interface ProcessEnv {
//       NODE_ENV: string;
//     }
//   }
// }

// // Mock window.fetch for API tests
// globalThis.fetch = vi.fn();

// // Mock ResizeObserver
// globalThis.ResizeObserver = vi.fn().mockImplementation(() => ({
//   observe: vi.fn(),
//   unobserve: vi.fn(),
//   disconnect: vi.fn(),
// }));

// // Mock IntersectionObserver
// globalThis.IntersectionObserver = vi.fn().mockImplementation(() => ({
//   observe: vi.fn(),
//   unobserve: vi.fn(),
//   disconnect: vi.fn(),
// }));

// // Mock SVG elements for graph visualization tests
// Object.defineProperty(window, 'SVGElement', {
//   value: class SVGElement extends HTMLElement {
//     getBBox() {
//       return { x: 0, y: 0, width: 100, height: 100 };
//     }
//   },
// });

// // Mock canvas for potential canvas-based visualizations
// HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
//   fillRect: vi.fn(),
//   clearRect: vi.fn(),
//   getImageData: vi.fn(() => ({ data: new Array(4) })),
//   putImageData: vi.fn(),
//   createImageData: vi.fn(() => ({ data: new Array(4) })),
//   setTransform: vi.fn(),
//   drawImage: vi.fn(),
//   save: vi.fn(),
//   fillText: vi.fn(),
//   restore: vi.fn(),
//   beginPath: vi.fn(),
//   moveTo: vi.fn(),
//   lineTo: vi.fn(),
//   closePath: vi.fn(),
//   stroke: vi.fn(),
//   translate: vi.fn(),
//   scale: vi.fn(),
//   rotate: vi.fn(),
//   arc: vi.fn(),
//   fill: vi.fn(),
//   measureText: vi.fn(() => ({ width: 0 })),
//   transform: vi.fn(),
//   rect: vi.fn(),
//   clip: vi.fn(),
// });

// // Mock file API for upload tests
// globalThis.File = class MockFile extends Blob {
//   name: string;
//   lastModified: number;
  
//   constructor(chunks: BlobPart[], filename: string, options?: FilePropertyBag) {
//     super(chunks, options);
//     this.name = filename;
//     this.lastModified = Date.now();
//   }
// } as any;

// // Simplified FileReader mock to avoid complex type issues
// globalThis.FileReader = vi.fn().mockImplementation(() => ({
//   result: null,
//   error: null,
//   readyState: 0,
//   onload: null,
//   onerror: null,
//   readAsText: vi.fn(function(this: any, _blob: Blob) {
//     setTimeout(() => {
//       this.result = 'mock file content';
//       this.readyState = 2;
//       if (this.onload) this.onload(new ProgressEvent('load'));
//     }, 0);
//   }),
//   readAsDataURL: vi.fn(function(this: any, _blob: Blob) {
//     setTimeout(() => {
//       this.result = 'data:text/plain;base64,bW9jayBmaWxlIGNvbnRlbnQ=';
//       this.readyState = 2;
//       if (this.onload) this.onload(new ProgressEvent('load'));
//     }, 0);
//   }),
//   readAsArrayBuffer: vi.fn(function(this: any, _blob: Blob) {
//     setTimeout(() => {
//       this.result = new ArrayBuffer(8);
//       this.readyState = 2;
//       if (this.onload) this.onload(new ProgressEvent('load'));
//     }, 0);
//   }),
//   abort: vi.fn()
// })) as any;

// // Mock URL.createObjectURL for file handling
// URL.createObjectURL = vi.fn(() => 'mock-url');
// URL.revokeObjectURL = vi.fn();

// // Mock FormData for file uploads
// globalThis.FormData = class MockFormData {
//   private data: Map<string, any> = new Map();
  
//   append(name: string, value: any) {
//     this.data.set(name, value);
//   }
  
//   get(name: string) {
//     return this.data.get(name);
//   }
  
//   has(name: string) {
//     return this.data.has(name);
//   }
  
//   delete(name: string) {
//     this.data.delete(name);
//   }
  
//   entries() {
//     return this.data.entries();
//   }
// } as any;

// // Mock WebSocket for real-time features (if added later)
// globalThis.WebSocket = vi.fn().mockImplementation(() => ({
//   send: vi.fn(),
//   close: vi.fn(),
//   addEventListener: vi.fn(),
//   removeEventListener: vi.fn(),
//   readyState: 1, // OPEN
// })) as any;

// // Set up default test environment variables
// if (typeof process !== 'undefined') {
//   process.env.NODE_ENV = 'test';
// }

// // Add custom matchers - import expect from vitest
// expect.extend({
//   toBeInRange(received: number, floor: number, ceiling: number) {
//     const pass = received >= floor && received <= ceiling;
//     if (pass) {
//       return {
//         message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
//         pass: true,
//       };
//     } else {
//       return {
//         message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
//         pass: false,
//       };
//     }
//   },
// });

// // Extend expect types
// declare module 'vitest' {
//   interface Assertion<T = any> {
//     toBeInRange(floor: number, ceiling: number): T;
//   }
// }