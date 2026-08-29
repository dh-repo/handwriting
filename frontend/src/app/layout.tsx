import type { Metadata } from 'next';
import './globals.css';
import { DocumentProvider } from '../context/DocumentContext';

export const metadata: Metadata = {
  title: 'Handwriting OCR & Prescription AI Workspace',
  description:
    'End-to-end handwriting recognition for messy cursive, medical prescriptions, and multi-page documents on Apple Silicon MPS and Vercel.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-indigo-500 selection:text-white">
        <DocumentProvider>{children}</DocumentProvider>
      </body>
    </html>
  );
}
