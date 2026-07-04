"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Upload, ListChecks, MessageCircle, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/", label: "Upload", icon: Upload, desc: "Ingest documents" },
  { href: "/jobs", label: "Jobs", icon: ListChecks, desc: "Track processing" },
  { href: "/chat", label: "Chat", icon: MessageCircle, desc: "Query knowledge" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed top-0 left-0 h-screen w-52 flex flex-col border-r border-sidebar-border bg-sidebar z-10">
      {/* Brand */}
      <div className="px-4 pt-6 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl gradient-brand flex items-center justify-center shrink-0 shadow-lg">
            <Layers size={15} className="text-white" strokeWidth={2} />
          </div>
          <div>
            <p className="text-sm font-bold tracking-tight text-gradient-brand leading-none">RAG Studio</p>
            <p className="text-[10px] text-muted-foreground mt-0.5 leading-none">v1 · Internal</p>
          </div>
        </div>
      </div>

      <div className="mx-3 h-px bg-sidebar-border" />

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 p-3 flex-1 mt-1">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground px-3 mb-2 mt-1">Navigation</p>
        {nav.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "group flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150",
                active
                  ? "bg-cyan-700/15 text-cyan-700"
                  : "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
              )}
            >
              <div className={cn(
                "w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-colors",
                active
                  ? "bg-cyan-700/20"
                  : "bg-transparent group-hover:bg-sidebar-border/60"
              )}>
                <Icon size={15} strokeWidth={active ? 2.2 : 1.8} />
              </div>
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer hint */}
      <div className="px-4 pb-5">
        <div className="rounded-xl bg-cyan-700/8 border border-cyan-700/15 px-3 py-2.5">
          <p className="text-[11px] font-medium text-cyan-700">Upload → Process → Chat</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">Three steps to search your docs</p>
        </div>
      </div>
    </aside>
  );
}
