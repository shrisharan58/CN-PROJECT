import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// --- Leaflet CSS and JS ---
// We add these here so they are loaded on every page,
// making the 'L' global object available for app/page.tsx.
import Head from "next/head";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CyberShield AI - Advanced Threat Detection",
  description: "Real-time AI-powered cybersecurity threat monitoring and detection system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        {/* ADD THESE TWO LINES FOR THE MAP */}
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.css"
        />
        <script
          src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.js"
          async
        ></script>
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased transition-all duration-300`}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}