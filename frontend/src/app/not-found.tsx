import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center p-8">
      <h2 className="text-2xl font-bold text-slate-100 mb-2">Document Not Found</h2>
      <p className="text-slate-400 mb-6">The requested document or page does not exist.</p>
      <Link
        href="/"
        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white font-medium transition-colors"
      >
        Return to Workspace
      </Link>
    </div>
  );
}
