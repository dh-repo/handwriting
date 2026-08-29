import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import NotFound from '@/app/not-found';

describe('NotFound 404 Route Component', () => {
  it('renders 404 header, descriptive message, and return home link', () => {
    render(<NotFound />);

    expect(screen.getByText('Document Not Found')).toBeInTheDocument();
    expect(screen.getByText('The requested document or page does not exist.')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /Return to Workspace/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/');
  });
});
