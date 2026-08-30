import type { Metadata } from 'next';
import './globals.css';
import { DocumentProvider } from '../context/DocumentContext';

const siteUrl = 'https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: 'Handwriting OCR & Prescription AI Workspace',
  description:
    'End-to-end handwriting recognition for messy cursive, medical prescriptions, and multi-page documents on Microsoft Azure Container Apps and Apple Silicon MPS.',
  alternates: {
    canonical: siteUrl,
  },
  openGraph: {
    title: 'Handwriting OCR & Prescription AI Workspace',
    description:
      'End-to-end handwriting recognition for messy cursive, medical prescriptions, and multi-page documents on Microsoft Azure Container Apps and Apple Silicon MPS.',
    url: siteUrl,
    siteName: 'Handwriting OCR AI Platform',
    type: 'website',
  },
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
