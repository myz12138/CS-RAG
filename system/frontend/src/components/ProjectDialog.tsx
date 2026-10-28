import { useEffect, useRef } from "react";
import { Icon } from "./Icon";
export function ProjectDialog({ open, name, setName, busy, connected, onClose, onSubmit }: { open: boolean; name: string; setName: (value: string) => void; busy: boolean; connected: boolean; onClose: () => void; onSubmit: () => Promise<void> }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { const element = dialog.current; if (open && element && !element.open) element.showModal(); else if (!open && element?.open) element.close(); }, [open]);
  return <dialog ref={dialog} className="project-dialog" onCancel={event => { event.preventDefault(); if (!busy) onClose(); }} onClick={event => { if (event.target === event.currentTarget && !busy) onClose(); }} aria-labelledby="new-project-title">
    <form onSubmit={event => { event.preventDefault(); void onSubmit(); }}><div className="dialog-heading"><div><span className="eyebrow">知识工作空间</span><h2 id="new-project-title">新建知识库项目</h2></div><button type="button" aria-label="关闭新建项目" onClick={onClose} disabled={busy}><Icon name="close" /></button></div><p>每个项目独立保存来源、知识图谱、文本索引与版本。</p><label className="field-label" htmlFor="project-name">项目名称</label><input autoFocus id="project-name" className="text-input" maxLength={100} required value={name} onChange={event => setName(event.target.value)} placeholder="例如：研发技术文档库"/><div className="dialog-actions"><button type="button" className="secondary-button" disabled={busy} onClick={onClose}>取消</button><button className="primary-button" disabled={busy || !connected || !name.trim()} type="submit">{busy ? "创建中…" : "创建项目"}</button></div></form>
  </dialog>;
}
