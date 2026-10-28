import { useState } from "react";
import { useWorkspace } from "./hooks/useWorkspace";
import { WorkspaceShell } from "./components/WorkspaceShell";
import { ProjectDialog } from "./components/ProjectDialog";
import OverviewView from "./pages/OverviewView";
import type { LibraryLanding } from "./types/workspace";
import KnowledgeBaseView from "./pages/KnowledgeBaseView";
import QueryView from "./pages/QueryView";
import SettingsView from "./pages/SettingsView";
import EvaluationView from "./pages/EvaluationView";
import TasksView from "./pages/TasksView";

export default function App() {
  const workspace = useWorkspace();
  const [createOpen, setCreateOpen] = useState(false);
  const [libraryLanding, setLibraryLanding] = useState<LibraryLanding>("sources");
  const { health, capabilities, projects, tasks, connection, activeNav, setActiveNav, projectName, setProjectName, selectedProjectId, setSelectedProjectId, versions, selectedVersionId, setSelectedVersionId, sources, versionDetail, versionContents, knowledgeVersionLoading, buildPreview, uploading, building, buildTask, buildSnapshots, creating, queryQuestion, setQueryQuestion, queryRunning, queryRecord, queryEvents, queryTaskProgress, queryTaskMessage, queryTraceLevel, setQueryTraceLevel, selectedConstraintId, constraintDetail, constraintLoading, evidenceDetail, evaluationSummary, modelConfig, buildConfig, queryConfig, modelChecking, modelCheckMessage, modelSaving, modelSaveMessage, buildSaving, buildSaveMessage, querySaving, querySaveMessage, handleCreateProject, refreshKnowledgeBase, handleOpenKnowledgeVersion, handleUpload, handleBuild, handleQuery, handleConstraintSelect, handleEvidenceSelect, handleModelCheck, handleModelSave, handleBuildConfigSave, handleQueryConfigSave } = workspace;
  return <WorkspaceShell workspace={workspace} onCreateProject={() => setCreateOpen(true)}>
    {activeNav === "总览" ? <OverviewView workspace={workspace} onCreateProject={() => setCreateOpen(true)} onOpenLibrary={landing => { setLibraryLanding(landing); setActiveNav("知识库"); }} /> : activeNav === "知识库" ? (
            <KnowledgeBaseView
              initialLanding={libraryLanding}
              projects={projects}
              selectedProjectId={selectedProjectId}
              versions={versions}
              selectedVersionId={selectedVersionId}
              sources={sources}
              versionDetail={versionDetail}
              versionContents={versionContents}
              buildPreview={buildPreview}
              uploading={uploading}
              building={building}
              buildTask={buildTask}
              buildSnapshots={buildSnapshots}
              knowledgeVersionLoading={knowledgeVersionLoading}
              onProjectChange={(projectId) => setSelectedProjectId(projectId)}
              onVersionChange={setSelectedVersionId}
              onUpload={(files) => void handleUpload(files)}
              onUseExisting={() => setActiveNav("多跳问答")}
              onPreview={() => { if (selectedProjectId) void refreshKnowledgeBase(selectedProjectId); }}
              onBuild={() => void handleBuild()}
              onOpenVersion={(versionId) => void handleOpenKnowledgeVersion(versionId)}
            />
          ) : activeNav === "多跳问答" ? (
            <QueryView
              projects={projects}
              selectedProjectId={selectedProjectId}
              versions={versions}
              selectedVersionId={selectedVersionId}
              question={queryQuestion}
              running={queryRunning || Boolean(queryRecord && ["pending", "running"].includes(queryRecord.status))}
              record={queryRecord}
              events={queryEvents}
              taskProgress={queryTaskProgress}
              taskMessage={queryTaskMessage}
              selectedConstraintId={selectedConstraintId}
              constraintDetail={constraintDetail}
              constraintLoading={constraintLoading}
              evidenceDetail={evidenceDetail}
              queryConfig={queryConfig}
              traceLevel={queryTraceLevel}
              onTraceLevelChange={setQueryTraceLevel}
              onProjectChange={setSelectedProjectId}
              onVersionChange={setSelectedVersionId}
              onQuestionChange={setQueryQuestion}
              onSubmit={() => void handleQuery()}
              onConstraintSelect={(constraintId) => void handleConstraintSelect(constraintId)}
              onEvidenceSelect={(evidenceId) => void handleEvidenceSelect(evidenceId)}
            />

          ) : activeNav === "设置" ? (
            <SettingsView health={health} capabilities={capabilities} modelConfig={modelConfig} buildConfig={buildConfig} queryConfig={queryConfig} checking={modelChecking} checkMessage={modelCheckMessage} saving={modelSaving} saveMessage={modelSaveMessage} buildSaving={buildSaving} buildSaveMessage={buildSaveMessage} querySaving={querySaving} querySaveMessage={querySaveMessage} onModelCheck={() => void handleModelCheck()} onModelSave={(update) => void handleModelSave(update)} onBuildSave={(update) => void handleBuildConfigSave(update)} onQuerySave={(update) => void handleQueryConfigSave(update)} />
          ) : activeNav === "评测" ? (
            <EvaluationView projects={projects} selectedProjectId={selectedProjectId} summary={evaluationSummary} onProjectChange={setSelectedProjectId} onOpenQuery={queryId => void workspace.handleOpenQuery(queryId)} />
          ) : <TasksView workspace={workspace} />}
    <ProjectDialog open={createOpen} name={projectName} setName={setProjectName} busy={creating} connected={connection === "connected"} onClose={() => setCreateOpen(false)} onSubmit={async () => { if (await handleCreateProject()) setCreateOpen(false); }} />
  </WorkspaceShell>;
}
