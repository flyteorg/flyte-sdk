# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

_Add changelog entries here as PRs merge; they move under a `## [vX.Y.Z] - TBA` heading at release time._

## [v2.8.1] - 2026-09-16

- durable model turns via Hermes llm_execution middleware (#1591)
- repair the volume examples against flyteplugins-union 0.10.9 (#1584)
- add JaxRun runtime for ClusteredTaskEnvironment (#1588)
- add Artifact.source_action_url (#1586)
- Expose the console URL of the run that produced an artifact (#1585)

## [v2.8.0] - 2026-09-14

- propagate --image to tasks in every environment, not just the first (#1566)
- correct Raises in download_wandb_run_dir and download_wandb_sweep_dirs (#1576)
- report a non-string memory/disk/shm against the field that was wrong (#1581)
- don't report a disk quota as an SDK crash (FLYTE-SDK-8D) (#1578)
- browser-use agent powered by Lightpanda (#1351)
- Add `flyte proxy app`: authenticated localhost proxy into a Flyte App (#1435)
- fetch the code a run executed (#1579)
- make --dryrun work for projects containing app environments (#1580)
- accept dry_run in flyte.deploy, deprecate dryrun (#1573)
- export Documentation from flyte (#1574)
- add --version to flyte run (#1575)
- add V5E to TPUType so TPU v5e is reachable (#1548)
- GPU faults are catchable typed exceptions (#1524)
- map disk to ephemeral-storage and skip shm in pod_spec_from_resources (#1545)
- reject a gpu resource that is not one of its declared forms (#1560)
- queue resource cap example (#1556)
- form-encoded deliveries, GitHub App installation tokens, installation events (#1541)
- interactivity, slash commands, notify helpers, and hitl-backed approvals (#1540)
- examples/apps: make WebSocket handshake header limits configurable (#1557)

## [v2.7.2] - 2026-09-08

- Env-var control of in-cluster auth mode (ExternalCommand without EAGER_API_KEY) (#1536)
- Add support for A2 GPU accelerator (#1554)
- clamp Backoff.compute_delay to the cap instead of overflowing (#1544)
- treat Resources(gpu=0) as no accelerator instead of raising (#1547)
- pooled Monty 0.0.22 runtime; apps mirror local sys.path (#1539)
- Claim new flyteplugins PyPI names at merge time, not release time (#1538)

## [v2.7.1] - 2026-09-04

- triggers without automation + flyte.run(trigger) to fire on demand (#1535)
- examples/apps: unpin streamlit so protobuf>=6 can resolve (#1530)
- auto-size multipart parts for Dir.from_local_sync uploads (#1462)
- clear error when the PKCE redirect URI has no host:port (FLYTE-SDK-7A) (#1422)
- give the signed upload URL a lifetime that covers the upload (FLYTE-SDK-5F) (#1363)
- surface third-party image builder failures as ImageBuildError (FLYTE-SDK-58) (#1369)

## [v2.7.0] - 2026-09-03

- Add flyte.app.Subdomain for deploy-time subdomain resolution (#1526)
- don't report a 405 from a non-Connect endpoint as an SDK crash (FLYTE-SDK-81) (#1525)
- bump the uv group across 5 directories with 6 updates (#1520)
- Add flyte.is_control_plane_available() and Run.first_failure() for agent edit-and-relaunch loops (#1522)
- add GPU metrics examples for the run details metrics tab (#1478)
- survive a broken pyOpenSSL install (FLYTE-SDK-7T) (#1488)
- receive SaaS webhooks, at flyte.extras.webhooks (#1512)
- Add flyteplugins-llamacpp: serve GGUF models with llama.cpp (#1521)
- auto-size multipart parts for File uploads (from_local_sync + hashed from_local) (#1480)
- run sync sandbox tools off the event loop so traced tools work in agent code mode (#1523)
- rebuild after source changes (#1508)

## [v2.6.13] - 2026-09-01

- Handle backoff with new error (#1500)
- Add task-level cache max age support (#1501)
- clear stale error.pb at the start of every clustered attempt (#1519)

## [v2.6.12] - 2026-09-01

- Add `flyte run hello` and `flyte serve hello`, first steps that need no files (#1496)
- introduce service-account on task environment (#1509)
- root action cache key honors `Literal.hash`, like sub-actions do (#1497)
- Log client retries: warning on RESOURCE_EXHAUSTED, debug otherwise (#1498)

## [v2.6.11] - 2026-08-28

- resume checkpoints across JobSet whole-set restarts in clustered tasks (#1499)
- correct Image.from_base() WORKDIR guidance (#1413)
- update gh workflow to publish packages automatically (#1495)
- Enable caching on the hf_model prefetch task (#1494)
- Refresh terminal phase check in tail logs (#1492)
- correct with_uv_project's install args and reset_root_logger, in Google style (#1493)
- backtick literals that were rendering as prose, and check for it (#1487)

## [v2.6.10] - 2026-08-26

- ssh sessions exit 1 on images with a /bin/false passwd shell (Spark) (#1491)
- preserve image build run link on image cache hits (#1481)
- `--type markdown` emits plain Markdown, and six RST literals the style check could not see (#1475)
- Add timeout on Logs.tail for terminal runs (#1486)
- add devbox gh codespace and google colab (#1482)

## [v2.6.9] - 2026-08-25

- async LocalDB connection never opened after sync-only init (#1485)

## [v2.6.8] - 2026-08-25

- Use flytectl config scopes in client credentials if configured (#1483)

## [v2.6.7] - 2026-08-25

- asyncio.run() works in sync tasks; sync-task calls from async must use .aio (#1472)
- remove reuse/oomer_reuse.py example (#1479)

## [v2.6.6] - 2026-08-25

- make flyte start devbox work on Windows (FLYTE-SDK-79, FLYTE-SDK-7B) (#1421)
- Retry `PERMISSION_DENIED` on first attempt without cached credentials to support PKCE bootstrap (#1474)
- report `declares_options` in `--type json` (#1471)
- hf_model docstring documents CLI flags as Python parameters (#1451)
- harden the CLI index builder after #1463 (#1469)
- add `flyte gen docs --type json` and record plugin provenance at registration (#1470)
- speculative decoding, cache-aware routing, and current engine pins (#1460)
- add Image.from_pixi_script() (#1468)

## [v2.6.5] - 2026-08-21

- detect plugin-provided CLI groups that have no callback (#1463)
- support new inputs alongside --recover, in the CLI and the API (#1466)

## [v2.6.4] - 2026-08-21

- Add parent filter for remote Action listall (#1465)
- update pyproject description (#1458)
- resolve $ref variants and preserve Optional in dataclass schema reconstruction (#1457)

## [v2.6.3] - 2026-08-20

- move the allow_missing_source_outputs argument from with_runcontext to rerun method (#1455)
- More verbose logging for keyring failures (#1454)
- Clarify rerun and recover behavior (#1452)

## [v2.6.2] - 2026-08-18

- implements flyte get devbox (#1449)
- bump the uv group across 2 directories with 2 updates (#1445)
- fix 0-byte directory placeholder download (#1448)
- Example that shows how we can cancel a running workflow and abort gra… (#1446)

## [v2.6.1] - 2026-08-17

- Add --devbox shortcut to 'flyte create config' (#1439)
- flytesdk ray plugin (#1233)
- count triggers deployed through deploy_task (#1436)
- count create_run from the SDK (#1437)
- Publish flyteplugins-otel to PyPI on release (#1428)
- keep native tool calling working with durable=True (#1430)
- remove validate-codecov-config workflow (#1433)
-  keep console URLs clickable when they exceed terminal width (#1434)
- bump actions to node24 majors, silence Node 20 deprecation warnings (#1426)
- Remove 'Try in Browser' badge from README (#1429)
- don't crash-report a non-Connect endpoint answering a Connect RPC (FLYTE-SDK-77, FLYTE-SDK-7A) (#1411)
- Make the release publish workflow rerunnable (#1425)

## [v2.6.0] - 2026-08-12

- resolve confirmed action-name instabilities (ENG26-831) (#1409)
- Make action watch resilient to transient stream interruptions (#1423)
- Add --plugin-config and --clustered options to flyte run python-script #1410 (#1418)

## [v2.5.20] - 2026-08-12

- omit empty main tab from report (#1420)
- don't report to Sentry while a test is running (FLYTE-SDK-73) (#1412)
- Record serving facts at prefetch, and make the prefetch task registerable (#1405)
- Add control-plane reporting for tracked runs (flyte run --tracked) (#1356)
- Add flyte.load_interactive_ctx() to restore task context in a debugger (#1416)
- Install code-server extensions in one invocation to avoid EEXIST race (#1417)
- install ms-python.debugpy and bump code-server/extensions (#1415)
- add action-name stability matrix tests (ENG26-831) (#1406)
- stream app logs in the CLI via flyte serve --follow (#1407)
- rs_controller: publish flyte_core to crates.io (#1408)
- Give artifacts an explicit kind discriminator (#1402)
- Flush a leaf task's report at task end (#1397)
- Send the resolved artifact version for artifact-bound app parameters (#1400)
- Resolving the kubeconfig path for devbox start command (#1399)
- Rename artifact Metadata.data to attrs, CLI --data flag to --attr (#1396)
- rs_controller: support condition actions and decoupled start/wait (#1401)
- Fall back to _F_PATH_REWRITE env var for actor tasks in rusty.run_task (#1384)

## [v2.6.0b0] - 2026-08-07

- clear error when the configured endpoint doesn't resolve (FLYTE-SDK-6Z) (#1387)

## [v2.5.19] - 2026-08-07

- strip inert reStructuredText from docstrings, and gate it in CI (#1376)
- Remove multi-tenant central mode from the MCP server (#1391)
- fix setting config image registry (#1392)
- make action names stable across code changes (#1373)
- Example for external published artifact triggers (#1390)
- Artifacts and Artifact-Triggers (#1362)
- reject RayJob TTL/shutdown options when a reuse policy is set (#1388)
- bump the uv group across 46 directories with 4 updates (#1386)
- opentelemetry tracing and grafana agent observability for agents (#1355)
- add termination grace period (#1379)
- multi-cluster example (#581)

## [v2.5.18] - 2026-08-05

- populate TaskTemplate.reuse_policy for reusable tasks (#1350)
- Warn when running flyte run from the home directory (#1380)
- cap mcp below 2.0 in the mcp extra (#1377)

## [v2.5.18b0] - 2026-08-04

- Update links in README for Flyte 2 (#1374)
- add example on how to register an abort callback (#1371)
- serialize concurrent code-bundle download/extract to prevent worker startup crashes in clustered tasks (#1370)
- add mcp central and tools (#1365)
- Expose task serialization as a public method (#1368)
- preserve zero-byte files in Dir/File downloads (#1366)
- update README for Flyte 2 GA announcement (#1367)

## [v2.5.17] - 2026-08-03

- add --recover (rerun) and --recover-from (run) (#1237)
- await local-mode map results instead of yielding bare coroutines (#1361)
- pass task_name when `Trigger.get_details` fetches details (#1349)
- Revert #1312 (#1357)
- add Action.get_report to pull reports from actions inside a run (#1352)
- Use SelectCluster before calling GetImage (#1345)
- use ruff to bundle lazy/conditional imports for loaded_modules (#1265)
- redact signed-URL credentials from upload error messages (FLYTE-SDK-6R) (#1339)
- skip files that vanish before being added to code bundle (#1195)
- clear error for missing .dockerignore in with_dockerignore (#1184)
- wrap docker buildx ls/create failures as ImageBuildError (#1167)
- render trigger automation without crashing in `flyte get trigger` (#1343)
- route macOS keychain access through /usr/bin/security to kill per-venv password prompts (#1300)
- Fix integration tests (#1341)
- make module-level syncify wrappers cloudpickleable (#1102)
- raise InitializationError for invalid auth mode (#1194)
- [Devbox] Extend devbox init timeout (#1337)
- Allow extra docker buildx flags via env var (#1338)
- bump the uv group across 4 directories with 3 updates (#1287)
- run twine outside the project venv when publishing plugins (#1336)

## [v2.5.16] - 2026-07-28

- filter pyqwest transport errors from Sentry (FLYTE-SDK-6H) (#1333)
- Pixi: don't inject `--locked` when extra_args sets `--frozen` (#1335)
- accept an ImageBuilder instance in _get_builder (FLYTE-SDK-61) (#1276)
- interactive_mode=False should still be respected in a IPython/Jupyter environment (#1293)
- Support plugin_config in TaskTemplate.override (#1334)
- publish constraints file (#1331)

## [v2.5.15] - 2026-07-28

- serialize optional None task defaults (#1332)
- add app substate showcase example (#1330)
- forward USER_LOG_LEVEL to task pods (#1262)
- add lance plugin (#1313)
- bump the uv group across 11 directories with 5 updates (#1322)
- support ReusePolicy — run Ray tasks on shared, reusable RayClusters (#1308)
- Add status filter to App.listall() and --status flag to flyte get apps (#1327)
- make MCP transport="stdio" actually serve stdio (#1319)
- type-check all examples with mypy and ty; add ty make target + pre-commit hook (#1298)

## [v2.5.14] - 2026-07-23

- Set default registry in build_default_image.py (#1326)
- sync local sys paths for deploy (#1321)

## [v2.5.13] - 2026-07-23

- Preserve DRA resource claims when overriding task resources (#1324)
- bump flyteidl2 to 2.0.29 (#1323)
- Feat/default registry push (#1304)
- Update relation fields (#1303)

## [v2.5.12] - 2026-07-21

- route Dir.download_sync through obstore-aware storage.get (#1320)
- [Feat] Add sentry metrics for DeployTask, DeployApp, and DeployTrigger (#1295)
- wrap any user-module import exception as ModuleLoadError (#1168)
- add pixi support via Image.with_pixi_project (#1309)
- Pin dask distributed (#1312)

## [v2.5.11] - 2026-07-20

- restore system CA trust on pyqwest 0.7+ (#1315)
- add flyte.remote.Run.get_report() to fetch run HTML reports (#1310)

## [v2.5.10] - 2026-07-17

- add crewai, langgraph, langchain, pydantic_ai, and hermes agent plugins (#1297)
- degrade gracefully when keyring package is missing (FLYTE-SDK-6N) (#1301)
- change download dir to cwd instead of home (#1311)
- nsight systems plugin (#1305)
- Network/handle deadline exceeded (#1296)
- add deprecation notice pointing to flyte-native conditions (#1307)
- use flyte-native conditions for HITL tool approval (#1306)
- store tokens in one keychain item to avoid double password prompt (#1292)

## [v2.5.9] - 2026-07-13

- restore original std streams while cloudpickling for version derivation (#1291)
- tolerate permission-denied files when copying image build context (FLYTE-SDK-6F) (#1283)
- json-raw CLI output tolerates non-serializable values like pathlib.Path (FLYTE-SDK-6G) (#1289)
- allow base registry override via config or FLYTE_IMAGE_R… (#1278)
- Add Trackio Plugin for flyte-sdk (#1281)
- make the Agent callback event stream multi-agent compatible (#1288)
- agents plugin (#1221)
- add remote tui (#1261)
- stage shell-task File inputs under NAMED_DIR (#1229)
- pin PORT env var so code-server binds to the correct port (#1286)
- populate RunSpec.related_to on rerun and in-container flyte.run (#1279)

## [v2.5.8] - 2026-07-07

- Revert "Bump connectrpc version and update usage" (#1284)
- fix a few core bugs (#1282)
- Bump connectrpc version and update usage (#1255)
- fix _Condition docstring example to use .aio() in async task (#1268)
- emit markdown-safe CLI reference (fenced examples, inline aliases) (#1269)
- Fix: SSH/VS Code debug entrypoint re-binds debug port (Errno 98) (#1273)
- Add voice customer-service example (Qwen/vLLM + switchable browser/Kokoro TTS) (#1267)
- Exclude union plugin (#1272)

## [v2.5.7] - 2026-06-29

- Fix: propagate sub-action error on parent recovery (USER error masked as SYSTEM) (#1270)

## [v2.5.7b1] - 2026-06-28

- replace with asyncssh so that no install at runtime (#1241)
- raise InitializationError when auth-config endpoint returns HTML (#1235)

## [v2.5.7b0] - 2026-06-26

- Classify task load EIO as system error (#1264)
- add clustered (JobSet) integration test (#1244)
- build flyte with pretend version in integration tests (#1259)
- bump the uv group across 12 directories with 5 updates (#1258)
- Filter runs by paused action (#1249)
- add CLI JSON parsing for File, Dir, and DataFrame in collections (#1257)
- fixed image classificaiton example for upstreaming huggingface changes (#1245)

## [v2.5.6] - 2026-06-25

- Route call_handler through Agent code mode (#1252)
- flyte run deployed-task ignores pinned version in input discovery (#1246)

## [v2.5.5] - 2026-06-25

- update uv lock files (#1256)
- Lower ceiling for connectrpc (#1254)
- typed output/input API on ActionDetails/Run (#1247)
- handle default_factory fields when reconstructing untagged outputs (#1243)
- Respect run_base_dir in UploadInputs (#1248)

## [v2.5.4] - 2026-06-23

- add support for flyte io in agents (#1242)
- signal unionai-docs to regenerate API docs on release (#1226)

## [v2.5.3] - 2026-06-22

- unify run/rerun/recover on one _Runner; add flyte.rerun (#1236)
- Clustered-task examples + end-to-end validation on the demo cluster (#1234)
- Adds support for queues to the CLI (#1238)

## [v2.5.2] - 2026-06-19

- SSH-into-task debug over WebSocket (#1230)
- Add labels to create run and list runs (#1228)
- repoint union plugin API-ref cross-link to union-plugin (DOC-1231) (#1227)
- populate code_bundle_uri in Task and App serialization (#1224)
- clustered (JobSet) restart — non-zero exit on failure + terminal-only error.pb (#1204)
- always install the local wheel for with_local_v2() (#1219)
- merge app resources into the primary container instead of overwriting (#1222)
- add file_input_layout to ContainerTask (#1220)
- Forward condition timeout, prompt_type, and webhook to ConditionAction proto (#1223)
- Create run with TaskId for lazy task (#1198)
- [Image Build] Enable setting platform in `from_base` and switch to `flyte` user only if exists (#1213)
- fix File, Dir, DataFrame cli parsing with collections (#1214)

## [v2.5.1] - 2026-06-15

- surface load-time errors in task UI instead of empty message (#1208)
- Fix secrets pagination (#1210)
- Fix trigger example (#1209)
- Update get action to show details (#1207)
- unprivileged allow_fuse() via FUSE device plugin + volume examples (#1205)
- Add 'flyte delete local-cache' command to clear the local cache directory (#1139)
- Add flyteplugins-redis: redis:// metadata storage via fsspec entry point (#1191)
- init_in_cluster: fall back to init_from_config when config file is present (#1182)
- merge requests/limits into head/worker pod_template instead of replacing it (#1196)
- merge task resources into pod template instead of overwri… (#1197)
- update flyte agents examples (#1201)

## [v2.5.0] - 2026-06-12

- Offload trigger inputs at registration (#1176)
- add nested dir example (#1192)
- guard ClusterAwareDataProxy against PKCE downgrade at cluster endpoint (#1185)
- filter ConnectError(UNIMPLEMENTED) from Sentry (#1177)
- Run action concurrency (#1190)
- Don't set start time when serializing task template (#1193)
- rename events to conditions (#1187)
- add ClusteredTaskEnvironment for distributed multi-node training (#1138)
- Downgrade path-rewrite fallback log to debug (#1186)
- Fix shell task (#1181)

## [v2.4.4] - 2026-06-08

- Add code_mode option to `Agent`, add `call_handler` to `@tool` decorator (#1179)
- include log level in console log lines (#1178)
- decouple MemoryStore from Agent (#1174)

## [v2.4.3] - 2026-06-07

- auto-size multipart parts so large uploads don't exceed the 10k-part limit (#1175)
- Small pydantic unit test (#1166)

## [v2.4.2] - 2026-06-06

- (bugfix) Overrides not applied to pods (#1173)

## [v2.4.1] - 2026-06-05

- [Devbox] Skip k8s context switch and kubeconfig update when kubectl not installed (#1150)
- PodTemplate capability helpers (allow_fuse / allow_nested_sandboxing, from_spec) + fix GPU accelerator in override() (#1170)
- coerce plain dicts into BaseModel/dataclass inputs for flyte.run (#1171)
- Notiifcation rules should be tuples (#1165)
- Local tui for events improved (#1164)
- include FlyteIgnore in default code-bundle ignores (#1162)
- add `flyte build --all` and a global `--no-progress` flag (#1153)
- snowflake connector runs in the data plane, not the control plane (#1159)
- filter transient network/timeout errors from Sentry (#1144)

## [v2.4.0] - 2026-06-03

- raise ImageBuildError for missing image source folder (#1148)
- wrap RuntimeError at user-module import as ModuleLoadError (#1146)
- raise InvalidImageNameError for unknown image ref in deploy (#1145)
- filter ConnectError(UNAVAILABLE/DEADLINE_EXCEEDED) from Sentry (#1137)
- honor Retry-After on upload 429/503 + raise default max backoff (#1136)
- include exception type in upload retry error when str(e) is empty (#1135)
- Syncify Agent.run (#1149)
- Fix template placeholders in notifications examples (#1143)
- expose current output name to type transformers (#1112)
- wrap KeyError at user-module import as ModuleLoadError (#1134)
- filter OSError(ENOSPC) from Sentry as user environment error (#1133)
- Fix integration tests (#1132)

## [v2.3.9] - 2026-06-02

- handle None record name in flyte log record factory (#1142)
- Queue examples (#1107)
- Add flyte-native Agent construct (#1121)

## [v2.3.8] - 2026-06-01

- set GPU extended_resources for Ray head and worker groups (#1126)
- Scope local cache by init config and add _cache_scope tests (#1131)
- Events API (#279)

## [v2.3.7] - 2026-05-29

- Fallback to constant start time and add lock around submit_sync (#1130)
- Update edit settings to remove unused string list setting  (#1117)
- Fixes recursive traces (#1128)
- Remove debug logs added for tracing stress test (#1127)
- consolidate image and bundle caches into LocalDB (#790)
- Better error message on upload failure (#1124)
- Removed heavy imports from direct path (#1123)
- filter ConnectError(Unauthenticated/PermissionDenied/FailedPrecondition) from Sentry (#1109)
- wrap uv export failures as ImageBuildError with stderr context (#1103)
- Revert block_network support in sandbox (#1120)
- add support for Union types with Field annotation in pydantic types (#1116)
- Pin default-dev-image PyPI fallback to flyte<{base_version} (#1122)
- Fix integratoin tests (#1118)
- [Controller] install rs controller from pypi if its dist folder does not exists (#1114)

## [v2.3.6] - 2026-05-26

- simplify flyte mcp (#1115)
- use OS DNS resolver in pyqwest transport (#1077)
- Support for RetryStrategy and Timeout types (#1019)
- Adds support for run_start_time to flyte.ctx() (#1100)
- surface user-module load errors as ModuleLoadError + filter via Sentry user-error set (#1094)
- fix pydantic/dataclasses default values at serialization time (#1096)
- add FlyteIgnore for excluding git-tracked files from code bundles (#1098)
- Remote builder support package name (#1101)
- [Controller] Support trace in rust controller (#1093)

## [v2.3.5] - 2026-05-21

- Remove leftover debug print (#1097)
- add clustered task pod entrypoint module (#1092)
- bump the uv group across 37 directories with 1 update (#1086)
- clarify base image USER/WORKDIR requirements on Image.from_base() (#1071)
- accept common cache behavior aliases (enable/on/off/...) (#1087)
- walk exception cause chain in Sentry user-error filter (#1078)
- surface missing src path as ImageBuildError in copy_files_to_context (#1075)
- wrap docker buildx CalledProcessError as ImageBuildError (#1082)
- raise InitializationError for missing project/domain (#1081)
- show distinct sys.modules keys in duplicate-env error message (#1074)
- raise actionable error when module file is outside root_dir (#1073)
- update readme again (#1091)
- update readme (#1089)
- raise clear error when flyte.TriggerTime is used as a task default (#1070)
- include underlying cause in upload signed-url error message (#1079)
- resolve _caller_frame to user code in subclasses and factories (#1088)
- Catch and retry refresh credential errors (#1090)
- [CI] Add Rust controller to default image (#1083)

## [v2.3.4] - 2026-05-19

- cap snowflake-connector-python <5 in snowflake plugin (#1084)

## [v2.3.3] - 2026-05-19

- add bio helper utility (#1055)
- guard against None responses in informer watch loop (#1080)
- gate Flyte blob streaming on model_path for vLLM and SGLang apps (#1076)

## [v2.3.2] - 2026-05-15

- escape bracket in read_file_if_exists debug log (#1072)
- Exclude external packages newer than 5 days old (#1069)

## [v2.3.1] - 2026-05-14

- bump vite from 5.4.21 to 8.0.12 in /examples/apps/vue_app in the npm_and_yarn group across 1 directory (#1068)

## [v2.3.0] - 2026-05-14

- Use SelectCluster for secrets client (#991)
- pin plugin wheel version via SETUPTOOLS_SCM_PRETEND_VERSION on release (#1067)

## [v2.3.0b0] - 2026-05-13

- fix publish CI (#1066)
- bump the cargo group across 1 directory with 3 updates (#1046)
- skip symlinks pointing outside source when bundling code (#1063)
- surface actionable error when docker is missing for image build (#1064)
- Simple Vue app example (#1026)
- bump urllib3 from 2.6.3 to 2.7.0 in /examples/ml/image_classification in the uv group across 1 directory (#1061)
- fall back to ASCII spinner when stdout can't encode unicode (#1058)
- Add get_logs example to custom connector (#1022)
- fix default values handling in the CLI for pydantic and dataclasses (#1059)
- retry upload on httpx.ReadError and other network errors (#1057)
- Add exclude-newer option to uv: delay new dependency releases by 5 days (#1060)
- filter InitializationError from Sentry (#1056)
- propagate USER_LOG_LEVEL to remote task and serve environments (#1034)
- remove repair_union_prev_checkpoint_uri SDK workaround (#1054)
- surface unpicklable deployments as ClickException (#1051)
- surface friendly ClickException when kubectl is missing in devbox (#1045)
- bump the uv group across 27 directories with 5 updates (#1053)
- surface user-module load errors as ClickException (#1050)
- Flyte MCP: native MCP to access flyte remote commands and more. (#1036)
- Add CodeModeAgent and AgentChatAppEnvironment in flyte.ai submodule (#1021)
- correct app deploy URL columns (#1049)
- filter DeploymentError/ImageBuildError from Sentry (#1052)
- Example of serving graphs in flyte (#1048)
- Added stress test (#1015)
- Rust-based controller (#488)

## [v2.2.4] - 2026-05-08

- surface devbox docker/kubeconfig errors as ClickException (#1041)
- coerce CopyConfig.src to Path in __post_init__ (#1040)
- stamp run/action context via LogRecordFactory (#1038)
- Allow overriding buildx builder via FLYTE_DOCKER_BUILDKIT_BUILDER_NAME (#1025)
- remove stale devbox container on restart (#1043)
- Update flyteidl2 version to newest 2.0.15 (#1044)
- don't report click.Abort/ClickException to Sentry (#1039)
- accept list of triggers/links in @env.task decorator (#1037)
- streamline shutdown handling in _serve function (#1032)
- Errors, exiting with 0, expects outputs (#1030)

## [v2.2.3] - 2026-05-04

- add user-facing logger as flyte.logger  (#1027)
- Use non-root flyte user by default in built images (#1016)
- Add vLLM + Claude Code integration example (#1017)
- resolve AppEnvironment caller frame across __post_init__ chains (#1023)

## [v2.2.2] - 2026-04-29

- update private_base_image example secret instructions (#988)
- Aborted runs will cause the controller to abort fast (#1020)

## [v2.2.1] - 2026-04-28

- Check docker availability before launching devbox (#1018)
- add HF dataset plugin (#992)
- update readme for devbox launch (#1009)
- Settings CLI and remote apis (#371)

## [v2.2.0] - 2026-04-27

- use cr.flyte.org registry for devbox images (#1014)
- add Dir support to code sandbox and codegen plugin (#1003)
- Correctly set the request id (#1013)
- bump ray from 2.54.0 to 2.55.0 in /plugins/ray in the uv group across 1 directory (#1008)
- [Devbox] Add Port 30081 for Apps Endpoint Serving (#1011)
- Resolve --builder default from flytectl config (#1007)
- Better Trace rendering in tui (#1002)
- add --builder flag to top-level flyte cli (#1004)
- resolve CodeBundleLayer in flyte build (#1001)
- forward api_key to per-cluster DataProxy session (#1000)
- Fix Task.listall() docstring: runs → tasks (#999)
- make fixes to python script cli (#998)
- Integrate Sentry SDK (#872)

## [v2.1.9] - 2026-04-21

- bump the uv group across 3 directories with 2 updates (#997)
- conditional cache example (#979)
- example of FastAPI and TokenBatcher in Flyte App environment (#961)
- Fast Embedding (#994)
- add hydra plugin (#965)

## [v2.1.8] - 2026-04-21

- rename flyte start demo to devbox (#993)
- Fix delete secret for scoped secrets (#995)
- block network access in the sandbox (#970)
- Add --gpu flag to `flyte start demo` (#989)
- Tri attention (#990)
- Include other files for fast-deploy in tasks and apps (#980)
- Sleep task type support in flytesdk (#978)
- Update demo cluster image and progress labels (#987)
- introduce FLYTE_IMAGEBUILDER_TASK_DOMAIN env variable (#985)
- add YAML representation to the report (#981)
- Faster runtime - upto 50% faster at runtime (#984)
- Actors - lower log line (#983)
- Add auth endpoint to create_session_config (#982)
- Update SDK to call Dataproxy.GetActionData (#953)
- Use TailLogs defined in dataproxy (#962)
- Add dataproxy client selector (#959)
- bump cryptography from 46.0.5 to 46.0.7 in /examples/genai/handoff in the uv group across 1 directory (#977)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/wandb (#870)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/codegen (#871)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/hitl (#873)
- Update uv.lock files in examples (#976)
- bump the uv group across 5 directories with 4 updates (#975)
- Bump cryptography from 46.0.3 to 46.0.5 in /examples/genai/handoff (#661)
- bump pygments from 2.19.2 to 2.20.0 in /examples/published-library/my-flyte-project (#869)
- bump pygments from 2.19.2 to 2.20.0 in /examples/uv_monorepo_guide/01_workspace_monorepo (#875)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/spark (#874)
- bump pygments from 2.19.2 to 2.20.0 in /examples/ml/ocr (#867)
- adds papermill plugin (#857)
- Entrypoint flag that helps with finding important tasks (#821)
- bring your own image in v2 (#863)
- add custom context example (#926)
- bump the uv group across 28 directories with 7 updates (#969)
- tag default image with local flyte version (#972)
- add disable_keyring config option to skip token caching (#971)
- adds omegaconf plugin (#924)
- enhance TaskEnvironment and Image.clone() docstrings (#964)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/anthropic (#936)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/databricks (#937)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/wandb (#938)
- bump pytest from 9.0.2 to 9.0.3 in /plugins/openai (#968)
- bump cryptography from 46.0.5 to 46.0.7 in /examples/ml/image_classification (#947)
- Fix duration of previous runs not reported in TUI (#958)
- bump pytest from 9.0.2 to 9.0.3 in /plugins/pandera (#967)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/mlflow (#963)
- Mitigate cross device issue (#960)
- Replace _is_flyte_default with _is_cloned on Image (#957)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/jsonl (#935)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/codegen (#946)
- bump cryptography from 46.0.6 to 46.0.7 in /plugins/openai (#948)
- Rename k8s context default from flytev2-sandbox to flyte-demo (#950)
- unwraps single `TaskGroup` failures in parallel reader (#951)
- Fix WantReadError in _bootstrap_ssl_from_server (#918)
- [Demo] prevent overwrite kubconfig cluster entry without CA (#949)
- Fix CLI table column truncation in flyte get secret (#894)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/openai (#923)
- Improve demo CLI UX with progress bar and graceful delete (#922)
- add custom_context to Trigger (#925)
- flyte serve: make Parameter more user-friendly (#914)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/pytorch (#911)
- bump transformers from 4.57.3 to 5.0.0rc3 in /examples/ml/image_classification (#920)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/bigquery (#868)
- clean-up: remove unneeded asyncify (#921)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/vllm (#919)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/ray (#912)
- Add custom connector app example (#913)
- Fix code bundle upload hanging indefinitely on network issues (#896)
- Fix insecureSkipVerify for corporate/private CA certs (#915)
- Use localhost registry when flyte endpoint is localhost (#893)
- [Demo cluster] Add flyte stop demo (#910)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/mlflow (#909)
- implement intratask checkpointing in the sdk (#885)
- Add codespell support with configuration and fixes (#853)
- bump pygments from 2.19.2 to 2.20.0 in /examples/project_structures/uv_project_local_dep/my_app (#877)
- bump aiohttp from 3.13.3 to 3.13.4 (#887)
- fix bug in Image.with_source_file not bumping image hash (#903)
- bump aiohttp from 3.13.3 to 3.13.4 in /examples/ml/image_classification (#891)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/codegen (#892)
- simple autoresearch example (#840)
- bump aiohttp from 3.13.3 to 3.13.4 in /examples/ml/ocr (#886)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/ray (#888)
- Update CreateRun to call UploadInputs first (#899)
- add google tag manager to flyte2intro app (#900)
- pin flyteidl2 to 2.0.11 (#905)
- bump litellm from 1.82.1 to 1.83.0 in /plugins/codegen (#904)
- guard against code_bundle being None in pre() (#901)
- preserve falsy default parameter values in task spec serialization (#902)
- Fix templates in notifications examples (#897)
- accept str as a valid Docker image path (#898)
- Fix Python 3.10 test hangs by updating parallel reader tests (#895)
- Update docs variant system: byoc+selfmanaged -> union (#860)
- Fix stale auth header accumulation on ConnectRPC retry (#890)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/databricks (#889)
- Filter non-user files from ImportError diagnostics (#881)
- Amend log (#884)
- bump pygments from 2.19.2 to 2.20.0 in /examples/uv_monorepo_guide/02_sibling_packages/my_app (#878)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/anthropic (#879)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/sglang (#876)
- bump pygments from 2.19.2 to 2.20.0 in /examples/published-library/my-task-library (#864)
-  fix: surface worker errors hidden inside BaseExceptionGroup in parallel reader (#883)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/pandera (#880)
- [Demo cluster] Skip cluster status check in dev mode (#861)
- Surface subtask errors instead of confusing TypeError in flyte.map (#882)
- bump pygments from 2.19.2 to 2.20.0 (#866)
- Replace GRPC with connectRPC (#844)
- bump cryptography from 46.0.5 to 46.0.6 in /plugins/openai (#862)
- add raw data path to apps (#740)
-  feat: respect .dockerignore in image cache key computation (#799)
- bump cryptography from 46.0.5 to 46.0.6 in /plugins/ray (#859)
- Remove union reuse lock (#858)
- Improved MLE agent (#854)
- bump pyasn1 from 0.6.2 to 0.6.3 in /plugins/mlflow (#825)
- [Demo cluster] Watch for demo cluster ready on start up (#855)
- bump requests from 2.32.5 to 2.33.0 in /examples/ml/ocr (#842)
- bump requests from 2.32.5 to 2.33.0 in /plugins/databricks (#843)
- bump requests from 2.32.5 to 2.33.0 in /plugins/wandb (#845)
- bump requests from 2.32.5 to 2.33.0 in /examples/project_structures/uv_project (#846)
- bump requests from 2.32.5 to 2.33.0 in /plugins/bigquery (#847)
- bump requests from 2.32.5 to 2.33.0 in /plugins/mlflow (#848)
- bump requests from 2.32.5 to 2.33.0 in /plugins/spark (#849)
- bump requests from 2.32.5 to 2.33.0 in /plugins/codegen (#850)
- bump requests from 2.32.5 to 2.33.0 in /examples/ml/image_classification (#851)
- bump requests from 2.32.5 to 2.33.0 in /plugins/openai (#852)
- implement sync traces (#831)
- Refactor localhost endpoint port override to use str.partition (#841)
- custom model served (#839)
- account for duplicate env deployment (#835)
- Simplify Dockerfile template virtualenv and UV_PYTHON configuration (#837)
- adds HF storage bucket example (#833)
- training checkpointing example (#834)

## [v2.0.12] - 2026-04-15

- add disable_keyring config option to skip token caching (backport #971) (#974)

## [v2.1.7] - 2026-04-14

- tag default image with local flyte version (#972)
- add disable_keyring config option to skip token caching (#971)

## [v2.1.6] - 2026-04-14

- adds omegaconf plugin (#924)
- enhance TaskEnvironment and Image.clone() docstrings (#964)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/anthropic (#936)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/databricks (#937)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/wandb (#938)
- bump pytest from 9.0.2 to 9.0.3 in /plugins/openai (#968)
- bump cryptography from 46.0.5 to 46.0.7 in /examples/ml/image_classification (#947)
- Fix duration of previous runs not reported in TUI (#958)
- bump pytest from 9.0.2 to 9.0.3 in /plugins/pandera (#967)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/mlflow (#963)
- Mitigate cross device issue (#960)
- Replace _is_flyte_default with _is_cloned on Image (#957)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/jsonl (#935)
- bump cryptography from 46.0.5 to 46.0.7 in /plugins/codegen (#946)
- bump cryptography from 46.0.6 to 46.0.7 in /plugins/openai (#948)

## [v2.1.5] - 2026-04-09

- Rename k8s context default from flytev2-sandbox to flyte-demo (#950)
- unwraps single `TaskGroup` failures in parallel reader (#951)
- Fix WantReadError in _bootstrap_ssl_from_server (#918)
- [Demo] prevent overwrite kubconfig cluster entry without CA (#949)
- Fix CLI table column truncation in flyte get secret (#894)

## [v2.1.4] - 2026-04-08

- bump pygments from 2.19.2 to 2.20.0 in /plugins/openai (#923)
- Improve demo CLI UX with progress bar and graceful delete (#922)
- add custom_context to Trigger (#925)
- flyte serve: make Parameter more user-friendly (#914)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/pytorch (#911)
- bump transformers from 4.57.3 to 5.0.0rc3 in /examples/ml/image_classification (#920)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/bigquery (#868)
- clean-up: remove unneeded asyncify (#921)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/vllm (#919)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/ray (#912)
- Add custom connector app example (#913)
- Fix code bundle upload hanging indefinitely on network issues (#896)
- Fix insecureSkipVerify for corporate/private CA certs (#915)
- Use localhost registry when flyte endpoint is localhost (#893)
- [Demo cluster] Add flyte stop demo (#910)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/mlflow (#909)

## [v2.1.3] - 2026-04-06

- implement intratask checkpointing in the sdk (#885)
- Add codespell support with configuration and fixes (#853)
- bump pygments from 2.19.2 to 2.20.0 in /examples/project_structures/uv_project_local_dep/my_app (#877)
- bump aiohttp from 3.13.3 to 3.13.4 (#887)
- fix bug in Image.with_source_file not bumping image hash (#903)
- bump aiohttp from 3.13.3 to 3.13.4 in /examples/ml/image_classification (#891)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/codegen (#892)
- simple autoresearch example (#840)
- bump aiohttp from 3.13.3 to 3.13.4 in /examples/ml/ocr (#886)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/ray (#888)
- Update CreateRun to call UploadInputs first (#899)
- add google tag manager to flyte2intro app (#900)
- pin flyteidl2 to 2.0.11 (#905)
- bump litellm from 1.82.1 to 1.83.0 in /plugins/codegen (#904)
- guard against code_bundle being None in pre() (#901)
- preserve falsy default parameter values in task spec serialization (#902)
- Fix templates in notifications examples (#897)
- accept str as a valid Docker image path (#898)
- Fix Python 3.10 test hangs by updating parallel reader tests (#895)
- Update docs variant system: byoc+selfmanaged -> union (#860)

## [v2.1.2] - 2026-04-01

- Fix stale auth header accumulation on ConnectRPC retry (#890)
- bump aiohttp from 3.13.3 to 3.13.4 in /plugins/databricks (#889)
- Filter non-user files from ImportError diagnostics (#881)

## [v2.1.1] - 2026-04-01

- Amend log (#884)
- bump pygments from 2.19.2 to 2.20.0 in /examples/uv_monorepo_guide/02_sibling_packages/my_app (#878)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/anthropic (#879)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/sglang (#876)
- bump pygments from 2.19.2 to 2.20.0 in /examples/published-library/my-task-library (#864)
-  fix: surface worker errors hidden inside BaseExceptionGroup in parallel reader (#883)
- bump pygments from 2.19.2 to 2.20.0 in /plugins/pandera (#880)
- [Demo cluster] Skip cluster status check in dev mode (#861)
- Surface subtask errors instead of confusing TypeError in flyte.map (#882)
- bump pygments from 2.19.2 to 2.20.0 (#866)

## [v2.1.0] - 2026-03-31

- Replace GRPC with connectRPC (#844)
- bump cryptography from 46.0.5 to 46.0.6 in /plugins/openai (#862)
- add raw data path to apps (#740)
-  feat: respect .dockerignore in image cache key computation (#799)
- bump cryptography from 46.0.5 to 46.0.6 in /plugins/ray (#859)
- Remove union reuse lock (#858)
- Improved MLE agent (#854)
- bump pyasn1 from 0.6.2 to 0.6.3 in /plugins/mlflow (#825)
- [Demo cluster] Watch for demo cluster ready on start up (#855)
- bump requests from 2.32.5 to 2.33.0 in /examples/ml/ocr (#842)
- bump requests from 2.32.5 to 2.33.0 in /plugins/databricks (#843)
- bump requests from 2.32.5 to 2.33.0 in /plugins/wandb (#845)
- bump requests from 2.32.5 to 2.33.0 in /examples/project_structures/uv_project (#846)
- bump requests from 2.32.5 to 2.33.0 in /plugins/bigquery (#847)
- bump requests from 2.32.5 to 2.33.0 in /plugins/mlflow (#848)
- bump requests from 2.32.5 to 2.33.0 in /plugins/spark (#849)
- bump requests from 2.32.5 to 2.33.0 in /plugins/codegen (#850)
- bump requests from 2.32.5 to 2.33.0 in /examples/ml/image_classification (#851)
- bump requests from 2.32.5 to 2.33.0 in /plugins/openai (#852)
- implement sync traces (#831)
- Refactor localhost endpoint port override to use str.partition (#841)
- custom model served (#839)
- account for duplicate env deployment (#835)
- Simplify Dockerfile template virtualenv and UV_PYTHON configuration (#837)
- adds HF storage bucket example (#833)
- training checkpointing example (#834)

## [v2.0.11] - 2026-03-24

- [Sandbox] Implement flyte v2 sandbox cli (#832)
- update anthropic api key secrets (#836)
- flyte.run using image with_code_bundle fails to discover all envs (#830)
- Fix infinite loop in log tail retry logic (#829)
- Add pandera plugin (#823)

## [v2.0.10] - 2026-03-23

- polars plugin parquet file is consistent with other dfs (#828)
- add host to wandb_config; snowflake: consolidate utils (#811)
- Skip image existence check for default flyte release images (#798)
- bump pyasn1 from 0.6.2 to 0.6.3 (#803)
- bump pyasn1 from 0.6.2 to 0.6.3 in /plugins/ray (#804)
- fix HITL plugin (#817)
- remove `block_network` flag from container task (#816)
- flyte.map concurrency control and large scale (#818)
- Notifications for triggers and runs (local and remote) (#414)
- Add missing obstore options for gcs (#815)
- normalize ".." in tar member names to prevent extraction failure (#814)
- File.download_sync does incremental download (#801)
- add flyte_map built-in for parallel task execution in code-mode workflow sandboxing (#812)

## [v2.0.10a0] - 2026-03-18

- test" g
- update
- add options for gcs
- Update init handling when API key is present in CLI  (#808)

## [v2.0.9] - 2026-03-18

- Enhance docstrings with detailed parameter documentation (#810)

## [v2.0.8] - 2026-03-17

- Update iteration of grpc metadata in auth interceptor (#807)
- bump pyasn1 from 0.6.2 to 0.6.3 in /plugins/bigquery (#802)
- prevent pre-release dependency resolution and sdist conflict in image builds (#806)
- bump cryptography from 46.0.3 to 46.0.5 in /examples/ml/image_classification (#797)

## [v2.0.7] - 2026-03-16

- monorepo examples and editable remote building fix (#795)
- run python script supports debugging, code bundling, output dir (#789)
- bump cryptography from 46.0.3 to 46.0.5 in /examples/ml/ocr (#787)
- bump pyjwt from 2.11.0 to 2.12.0 in /plugins/snowflake (#793)
- bump pyjwt from 2.11.0 to 2.12.0 in /plugins/codegen (#794)
- bump pyjwt from 2.11.0 to 2.12.0 in /plugins/openai (#796)
- mlflow integration (#767)
- Add (optional) `prek` for precommit hooks (#791)
- bump black from 25.12.0 to 26.3.1 in /examples/ml/ocr (#785)
- replace verbose docstrings with short descriptions in File, Dir, and DataFrame JSON schemas (#786)
- Add example for reusing a base image with existing UV_PYTHON venv (#762)
- add NonRecoverableError to skip retries on terminal failures (#780)
- bump tornado from 6.5.4 to 6.5.5 (#782)
- bump tornado from 6.5.4 to 6.5.5 in /plugins/dask (#783)
- Example test workflows to scale test image build and load (#774)
- fixes codegen plugin & CLI docs generation (#784)
- update codemode sandbox due to monty breaking change (#779)
- publish gemini plugin (#781)
- add pip options to Image.with_requirements() (#778)
- fix tool builder agent (#777)
- set ci plugin publishing fail-fast: false (#776)

## [v2.0.6] - 2026-03-11

- publish non-prerelease plugins (#775)

## [v2.0.5] - 2026-03-11

- skip auth interceptors when insecure=True (#761)
- Support for running arbit scripts (#773)
- add --debug flag and debug run context (#772)
- Make flyte2 panel app mobile-friendly (#769)
- Fix OpenAI and Anthropic plugin examples (#771)
- codegen plugin (#704)
- Add module docstrings and improve class/field docs for BigQuery, Databricks, Anthropic, Gemini plugins (#768)
- Overwrite cache on image build with force=True (#752)
- check plugin uv.lock files are up-to-date (#763)
- Add Google Gemini plugin for Flyte (#632)
- Add --plugin-variants flag to flyte gen docs (#764)
- Update flyte panel app: add idle time parameters (#759)
- update plugin uv lock files (#760)

## [v2.0.4] - 2026-03-06

- Add grpc headers when using Actions service (#753)
- auto-discover plugins for unit test matrix (#757)
- always use .aio() for Flyte tasks in FunctionTool.execute()   (#758)
- add support for `AWS_CONFIG_FILE` env vars (#750)
- More panel flyte 2 updates (#751)
- add anthropic plugin (#756)
- update plugins ci tests and publishing (#755)
- You can create project from cli now (#754)
- Increase code coverage to ensure user-facing api remains consistent (#749)
- Download file in parallel (#460)
- distributed training with eval (#735)
- Update pytorch examples (#746)
- add get_logs to remote.Run and remote.Action  (#748)
- accept enum names as valid CLI inputs in EnumParamType (#747)
- README.md update (#745)

## [v2.0.3] - 2026-03-03

- jsonl plugin (#728)
- Resuable tasks are not currently debuggable (#744)
- Fix(deploy): improve duplicate environment error message to hint at --root-dir (#707)
- add with_code_bundle to Image (#741)
- allow with_source_file to accept a list of paths  (#724)
- add phase runtimes to action details (#743)
- Support vim keys j/k navigation in status dropdown menu (#736)
- update panel flyte 2 app (#730)
- initialize config in TaskPerFileGroup commands (#742)
- MLE agent (#715)
- copy editable dep pyproject.toml files and compute common source_dir (#719)
- add project/domain/created_at/updated_at filters to Run.listall and Action.listall (#738)
- embed image build run URL in TaskMetadata for clickable image URI (#731)
- Add vim keys j,k navigation support for TUI menu (#727)
- local runs and tui should respect disable run cache (#732)
- Dynamic Batcher: Useful for dynamically batching onto a constrainted resource like gpus (#734)
- skip .venv and ignored directories in load_python_modules (#733)
- add support for local retries, support tui (#729)
- resolve include paths relative to app dir when bundling for flyte serve (#723)
- Example: add async training with periodic eval and early stopping exa… (#721)

## [v2.0.2] - 2026-02-26

- skip ignore files inside standard-ignored directories (#720)
- prevent double initialization logging in CLI run command (#722)
- stateless code sandbox (#658)
- human in the loop plugin (#657)
- adds `guess_python_type` to PydanticTransformer (#711)
- Update Ignores behavior (#717)
- add request_timeout support to AppEnvironment (#716)
- align `File.from_local` and `File.from_local_sync` file name (#713)
- route async Flyte task execution through task.aio() (#714)
- use correct field name for app metrics protocol (#712)
- Stress fanout (#710)
- Bug Fix: pod template should be carried over in replace: (#709)
- accept enum names as valid CLI inputs in EnumTransformer (#708)
- Fix shell injection of version-constrained pip package specs in dockerfile (#706)
- Update spark plugin to stable release >= 2.0.0 (#705)
- Panel app examples: simple calculator, Flyte in-browser experience (#701)
- Fix(anthropic): route tool input_schema through Flyte type engine (#702)

## [v2.0.1] - 2026-02-23

- Add S3 virtual-hosted-style addressing support for S3-compatible backends (#703)
- reverse priority of path entries (#692)
- Pytorch elastic hanging in case of cuda oom or non-rank0 exceptions (#698)
- H100 mig partitions (#700)
- [BUG]Fix ref task shortname override error (#699)
- Orchestration sandbox. (#667)
- Adds retry interceptor (#696)
- Bump cryptography from 46.0.3 to 46.0.5 in /examples/published-library/my-task-library (#679)
- Bump cryptography from 46.0.3 to 46.0.5 in /examples/published-library/my-flyte-project (#680)
- Add FlyteWebhookAppEnvironment to flyte.app.extras (#669)
- app spec merges image and pod template (#694)
- cli common to dict when available (#693)
- address empty app conditions list (#691)
- Code bundling using posix-style paths (#690)
- fix typeddict example for python<3.12 (#689)
- Example for cloning images using a script (e.g. in CI) (#685)
- Minor edit to readme (#688)

## [v2.0.0] - 2026-02-18

- Tuple fix (#684)

## [v2.0.0b60] - 2026-02-17

- remove check for parent dir (#683)
- resolve root (#682)

## [v2.0.0b59] - 2026-02-17

- Update CI - use github.ref_name instead of env var (#681)

## [v2.0.0b58] - 2026-02-17

- [Breaking] Add extendable parameter to Image class (#668)
- Play with actions service (#673)
- Fix lookup_image_in_cache for ref_name images (#670)
- Generalize custom type detection in nested Pydantic models (#664)
- Clean CLI output, no clobber, special status logger (#672)
- Improve clone existing image example for multi-env (#678)
- Fix __all__ typo in config/_config.py (#677)
- Support for untyped lists (#671)
- Implement user-facing `HashFunction`  (#666)
- Add support for .flyteignore file (#663)
- Add nested group example to grouping.py (#665)
- Bump pillow from 12.0.0 to 12.1.1 in /examples/ml/ocr (#653)
- Bump pillow from 12.0.0 to 12.1.1 in /examples/ml/image_classification (#654)
- skip modules that are not filepaths (#660)
- Fix for blob example (#659)
- Tests for Flyte File, Dir, DataFrame in Pydantic (#651)

## [v2.0.0b57] - 2026-02-11

- Local persistence does not need endpoint (#656)
- Fix Literal string serialization (#655)
- Add support for cloning an existing image (#652)
-  Bring determinism to your programs (#599)
- Better local apps (#649)
- Render dataframe example (#650)
- Bump cryptography from 46.0.4 to 46.0.5 (#648)
- Fix BigQuery connector for protobuf VariableMap changes (#639)
- Use enum names instead of values in literals (#618)
- Added support to handle nested Pydantic types (#640)
- add subdomain name to n8n flyte webhook app (#646)
- check if app spec is updated on re-deploy (#642)
- update n8n - require auth (#644)
- Add current_project helper and update examples (#645)
- Unit tests for list_files (#643)
- Add example of django app (#641)
- add n8n example (#591)

## [v2.0.0b56] - 2026-02-09

- Tasks cannot be nested within a trace! (#636)
- Import optimization improved (#638)
- Bump protobuf from 6.33.2 to 6.33.5 in /examples/ml/image_classification (#637)
- add anthropic agent deep research example (#635)
- Support local app serving (#628)
- Persistence of local runs (#627)
- fix tui output format to render newlines (#634)
- remove wandb uvlock (#626)

## [v2.0.0b55] - 2026-02-06

- Add integration tests for all examples (#543)
- Large IO Writing should close buffer (#625)
- Remove BigQuery handlers from dataframe module (#624)
- snowflake connector: add support for DF conversion, batch inputs, and update docstrings (#584)
- Added support for Enum types in Pydantic models (#622)
- Add Anthropic Claude plugin for Flyte (#612)
- Add first-class support for typeddict (#613)
- Bump protobuf from 6.33.0 to 6.33.5 in /examples/published-library/my-flyte-project (#623)
- fix df local sync in nested types (#620)
- Add polars lazyframe example with big data (#614)
- Introducing TUI: Flyte local runs (#621)
- Fix .egg-info directories not being ignored during file copy (#617)
- dynamic DAG generation from yamls (#619)
- Unpin unionai-reuse in examples (#616)
- [Feat] Typed interface migration (#558)
- Fix BigQuery credentials parsing and secret key typo (#606)
- Remove get_cwd_editable_install function (#615)
- Add first-class support for tuple and NamedTuple via pydantic (#597)
- default to single run for multi-node (#608)
- Bump protobuf from 6.33.0 to 6.33.5 (#609)
- Bump protobuf from 6.33.2 to 6.33.5 in /plugins/wandb (#610)
- Bump protobuf from 6.33.2 to 6.33.5 in /examples/ml/ocr (#611)

## [v2.0.0b54] - 2026-02-04

- Fixes a potential subtle race condition in call-seq-generation (#607)
- Fix duplicate secrets in image build commands (#604)
- Bump protobuf from 6.33.0 to 6.33.5 in /examples/published-library/my-task-library (#603)
- Remove flyteidl (#523)
- [Fix] .dockerignore pattern matching (#550)
- Disable rich formatting (spinner, colors) for json and table-simple output formats (#587)
- fix df local to remote (#594)
- Fail lint job when make fmt changes files (#602)
- Split connectors into separate PyPI packages (#595)
- Skip image building for Snowflake and BigQuery connector tasks (#559)
- Storage config nits (#592)
- add support for distributed training (#600)
- Bump protobuf from 6.33.2 to 6.33.5 in /examples/genai/handoff (#601)
- [Feat] use flyteidl2 dataproxy (#526)
- Make uvlock optional in UVProject (#596)
- Add example workflow with handling for UI-canceled actions in fanout group (#589)
- Add helpful error messages for list validation in Layer classes (#593)
- Add RequiresOption for CLI option dependencies (#586)

## [v2.0.0b53] - 2026-01-29

- during deployment tasks use correct image (#590)
- Creates a programmatic abort test (#588)
- [Fix] Make batch size configurable in Dir `from_local` (#585)
- Bump urllib3 from 2.6.2 to 2.6.3 in /examples/ml/ocr (#583)
- Bump python-multipart from 0.0.21 to 0.0.22 in /examples/ml/image_classification (#576)
- Improve error message when no files found to bundle (#582)
- Add wait in image builder protocol (#567)
- Use separate thread for running sync tasks (#565)
- Update polars examples (#579)
- Remove org as a requirement. (#577)
- Support uv editable installs; fix cwd editable detection and secret mounts (#560)
- Delete Apps (#580)

## [v2.0.0b52] - 2026-01-26

- Add link type to the bigquery link (#557)
- Support VIRTUAL_ENV override in UV and Poetry project installs (#561)
- Bump marshmallow from 3.26.1 to 3.26.2 in /examples/genai/handoff (#575)
- add wandb to gh wf (#566)
- Bump aiohttp from 3.13.2 to 3.13.3 in /examples/ml/image_classification (#574)
- Bump urllib3 from 2.6.2 to 2.6.3 in /examples/ml/image_classification (#573)
- Bump urllib3 from 2.5.0 to 2.6.3 in /examples/project_structures/uv_project (#554)
- Bump pyasn1 from 0.6.1 to 0.6.2 (#556)
- Bump marshmallow from 3.26.1 to 3.26.2 in /examples/published-library/my-task-library (#555)
- Bump filelock from 3.20.1 to 3.20.3 in /examples/ml/image_classification (#572)
- Bump filelock from 3.20.1 to 3.20.3 in /examples/ml/ocr (#571)
- Bump aiohttp from 3.13.2 to 3.13.3 in /examples/ml/ocr (#570)
- Bump urllib3 from 2.3.0 to 2.6.3 (#564)
- make fmt (#569)

## [v2.0.0b51] - 2026-01-26

- build polars plugin (#568)

## [v2.0.0b50] - 2026-01-23

- Implement polars plugin, improve overall dataframe developer experience (#527)
- [Fix] Gitignore not working when building code bundle (#551)
- add snowflake connector (#538)
- Rename variables in wandb plugin docstring examples (#552)
- wandb proto docs (#547)

## [v2.0.0b49] - 2026-01-21

- flyte serve should deploy dependent apps (#549)
- add support for private urls in AppEndpoint (#548)
- Stop truncating App URLs, include console deployment link too (#546)
- Fix vllm app example post release (#544)
- Fix `from_ref_name` (#545)
- adding checks for mandatory options org,project and domain (#536)
- add tests for auto-uploading (#541)
- add support for downloading w&b logs in the plugin (#542)
- Init pass through automatically uses env endpoint (#540)
- Support Upload local files, directories and dataframes to remote (#533)
- update images in vllm and sglang examples (#517)
- wandb plugin (#421)
- run sync app.server in a new event loop (#529)
- FIX update image classification example (#537)
- Small fixes for run_per_second example (#534)
- Dataframe passing example local and remote.  (#532)

## [v2.0.0b48] - 2026-01-10

- Contextual Request ID for flyte requests (#531)
- Fast imports (#525)
- Use execution context variables in ActionID template (#530)
- [breaking] Local run outputs are action outputs now (#524)
- add action abort support and CLI command (#520)
- Add log links for the root action (#522)
- ActionAbortedError is raised (#521)

## [v2.0.0b47] - 2026-01-08

- Check config file by click (#518)
- Removed ReferenceTaskError and RemoteTaskError (#519)

## [v2.0.0b46] - 2026-01-07

- remove publishing of plugin images (#516)
- Fix Passthrough authenticator and FasAPI middle for passthrough auth (#512)
- adding config file exception (#513)
- Add support for sglang 0.5.7, add plugin image building to ci (#514)
- combining pkce enum types (#508)
- Better implementation of flyte.init_from_api_key (#510)
- Pass through initialization and auth passing (#505)
- Fix vllm plugin import, pin vllm and sglang version in default images (#509)

## [v2.0.0b45] - 2026-01-06

- Extract correct app module name (#506)
- add unit tests for app deploy code bundling (#507)
- fix deploy app (#504)

## [v2.0.0b44] - 2026-01-05

- compileall in the local builder (#503)
- Image Builder Plugin (#502)
- make sure app versions are consistent with same include files (#501)
- Update vllm and sglang plugin dependencies (#500)
- add detailed explaination about the datatype of the missing parameter… (#498)
- Support apps in notebooks, add examples (#499)
- Moving dataframe extension to a separate place (#495)
- Document OCR example (#483)
- Remote entity API docs (#497)
- adding clarity to run method with breaking it into more managable parts (#496)
- common.initialize_config used instead of local initialize_config.Remo… (#494)
- library example, should use release flyte (#493)
- Better error handling in types (#492)

## [v2.0.0b43] - 2026-01-02

- Eval framework example (#287)
- Add support python 3.14 (#465)
- Update logging behavior (#474)
- Add a simple stopwatch (#455)
- Published library example (#275)
- Wikipedia sentence transformers update (#407)
- Mock exit for tests (#491)
- Silence grpcio polling when using non-Flyte event loops (#467)
- Fix Python 3.14 compatibility issues (#490)
- Added run-project/domain (#486)

## [v2.0.0b42] - 2025-12-31

- Named outputs (#485)
- Example of parallel processing deltalake (#481)
- Image classification update version flyte (#482)
- Do not import prefetch at flyte level (#479)

## [v2.0.0b41] - 2025-12-26

- Serving improvements (#478)
- Improve logging (#286)
- Runtime optimized (#475)
- Add wandb link (#397)
- Connector App (#470)
- add pdf text extraction example (#471)
- bind appenv parameters to server, startup, shutdown methods (#472)
- Update ReferenceTaskError -> RemoteTaskError on examples (#469)
- rename mention of "Reference Task" -> "Remote Task" (#468)
- Reference task error should be raised when not found (#466)
- Implement app_env.server, on_startup, on_shutdown decorators and pickled apps (#449)
- Cleanup examples (#464)
- Fixed run cli handling (#462)
- fix image classification index.html file read (#463)
- Cleanup items (#461)
- Image classification end to end example - Train, Serve and batch infer (#457)
- Add remote lazy validation example (#459)
- Remote tasks are handled better (#458)
- Example of unit testing (#456)
- fix config init (#453)
- Rename flyte.app.Input to flyte.app.Parameter (#454)

## [v2.0.0b40] - 2025-12-19

- GB10 added to sdk (#452)
- Init from config can override project/domain (#451)
- Fix refresh_token path for PKCE (#450)
- Use gcs as volumes through sidecar on gke (#448)
- Small tweaks (#446)
- Add vllm and sglang (#445)

## [v2.0.0b39] - 2025-12-18

- de-duplicate files in app code bundling (#444)
- Wait is now stateful (#443)
- Phase consolidation (#441)
- add default images for sglang and vllm (#442)
- Deploy improvements (#440)
- Fix PKCE to read the entire callback request properly (#439)
- Log relative path in FastAPIEnvironment module extraction. (#438)
- add support for directories and glob paths in app include (#437)
- Make `flyte prefetch hf-model` resource-related args more consistent (#433)
- add example for single script streamlit and python app (#436)
- rename prefetch arg: s3_path -> obstore_path (#428)

## [v2.0.0b38] - 2025-12-15

- Working example of multi folder structure with just pythonpath (#430)
- Add flyte prefetch verb with hf-model noun to prefetch huggingface models (#418)

## [v2.0.0b37] - 2025-12-14

- [Bug] Nested schema type detection for reference tasks does not work (#426)
- Update connector idl (#409)
- Add retry logic to file upload function (#415)
- stress-test: parallel image builds (#416)
- iceberg for production (#419)
- S3 cannot do update same object too many times (#420)
- Add keep alive option (#417)

## [v2.0.0b36] - 2025-12-11

- Add idle timeout to code server instead of relying on heartbeat file (#413)
- Improved Module loading speed and imports (#412)
- Change default for from_uv_script to accept prereleases (#411)
- Replacing bytes with chuck as keywords should not be shadowed (#410)
- overwrite existing file (#408)
- Add SGLangAppEnvironment (#399)
- Error handling fix (#404)
- adding additional files and folders to be ignored (#405)
- Fixing delayed inputs (#403)
- Improving recsys (#402)
- Example of an Agent handoff system with Apps and workflows (#400)
- constrain Python version to <3.13 (#401)
- Add integration tests (#154)
- add optional int collection example (#391)
- Moved Iceberg to data processing and fixed script (#398)
- Increase limit for oauth reader (#396)
- basic streaming reduce example (#394)
- API Key detection/use (#390)
- Update uv.lock for connector (#389)
- Implement VLLMAppEnvironment and SafeTensorStreaming (#387)
- support default ref name in Image.from_ref_name (#388)

## [v2.0.0b35] - 2025-12-08

- Install local project in uv script (#372)
- Improve path handling in copy_files_to_context (#378)
- pass --image CLI args to init in recursive deploy mode (#386)
- Update dependencies for embed_wikipedia example (#385)
- Add `input_values` to override app env input values in `with_servecontext` (#375)
- [Feature] Add task git repo host uri when deploying task (#335)
- Add c0 command for connector (#374)
- Update `flyte gen` to detect plugins (#384)
- Example that shows how to create an A/B/n test with App calling apps (#381)
- Large number of concurrent runs stress test (#383)
- Add H200 partitions (#380)
- Websocket example (#373)
- Adding serve to a command (#370)
- Databricks connector (#330)

## [v2.0.0b34] - 2025-12-02

- Add description to task deploy (#314)
- Prep for new release b34 (#368)
- fix run and syncify typing (#366)
- Added H200 to available devices (#367)
- App deploy and serve refactored. with_servecontext (#364)
- Embedding recsys (#360)
- [Refactor] Use ActionPhase from common phase proto (#361)
- update app replica range to (0, 1) (#363)
- flyte serve and deployment updates (#339)
- Improve CLI doc gen (#362)
- Minor updates (#356)
- Add support tool-uv-index in the remote builder (#357)
- set deref_symlinks to True (#359)
- Support custom image URIs as strings in TaskEnvironment (#348)
- Add custom context example (#355)
- Iceberg example using pyarrow tables (#340)
- Add support for `--image` to deploy (#352)
- Kyle/sdk/feat/add version info (#351)
- Call Apps from tasks. (#349)

## [v2.0.0b33] - 2025-11-20

- Introduce _U_INSECURE env var (#347)
- skip final report flush when no report is available (#344)
- pipe subdomain and custom domain to app idl (#346)
- Revert "Trigger time (#324)" (#345)

## [v2.0.0b32] - 2025-11-18

- Add bigquery extra to flyteplugins connector (#334)
- Comparison to s5cmd (#292)
- fix pythonpath deployment pattern example (#342)
- Increase PyPI publish wait time to 5 mins (#333)
- supporting the file_name parameter in the new_remote function (#337)
- Remove uv dev-dependencies from pyproject.toml (#332)
- Standardize init behavior across most examples (#338)
- Alpha support: Flyte apps. Serve models, streamlit apps and much more! (#212)
- Dataclass blob input example (#336)
- Pass image builder project-domain scope and cache lookup scope (#327)

## [v2.0.0b31] - 2025-11-10

- Allow prerelease uploads to PyPI  (#331)

## [v2.0.0b30] - 2025-11-10

- spark transformer (#289)
- Update trigger_serde (#329)
- Add example showing how to use time zones with cron schedule triggers (#326)
- Fix flyte serve cli (#328)
- Build flyte connector image (#321)
- Don't set default dst in with_source_folder (#325)
- Scope image build runs to the same target project-domain (#322)
- Trigger time (#324)
- Don't use local file keyring when not in ipython (#323)

## [v2.0.0b29] - 2025-11-06

- Add a lightweight JSON formatter (#320)
- Skip listing relative module files (#317)
- Set root handler level to debug (#319)
- Add task examples for various input types (#318)
- Type plugins (#306)
- Allow relative root dir (#316)
- remove trigger_time as default argument in triggers (#315)
- Change setup to sync in dask plugin (#307)
- [ImageBuilder] Add project_install_mode arg to Poetry layer in Image spec (#308)

## [v2.0.0b28] - 2025-10-31

- Use 3.13 to publish, not .x (#312)

## [v2.0.0b27] - 2025-10-31

- Minor update to _file_is_in_directory (#310)
- Bundle will follow symlinks (#309)
- Fix pyproject package deployment pattern readme (#305)
- Support CLI plugins (#303)
- introduce timezone for CRON (#260)
- update ml gridsearch example (#304)

## [v2.0.0b26] - 2025-10-29

- Connector Interface & Bigquery Connector (#209)
- Support for insecure endpoints (#299)
- Update creation of image pull secrets (#302)
- Add addition attributes in RunSpec (#217)
- Add missing allowed neuron quantities (#301)
- Add SchedulerPlugin for code bundle download (#294)
- Fast register uv workspace (#283)
- Use get_type_hints for param and return types (#295)
- Improve the error message when the image is not found in the image cache (#298)
- Incorrect Trigger type-hint and refactor module extraction (#297)
- Fix error message formatting for clickable build link (#291)
- fix sync map (#293)
- Support for installed function module extraction (#290)
- ImportError Diagnostics (#288)
- Updating listSecrets to have default page size of 10 (#285)
- Improve excepthook (#284)
- Unit testing is natural (#282)
- Slack bot faster (#281)
- Slack bot demo, that is crash-proof (#280)

## [v2.0.0b25] - 2025-10-17

- Default project_install_mode to dependencies_only (#277)
- create all types example (#160)
- Fixes logging of error path (#276)
- UV project best practice (#216)
- add `context` (#254)
- Add debuggable support to tasks and plugins (#240)
- Trigger update (#274)
- Fix mypy tasks (#265)
- Pyproject package Example (#273)
- make fmt (#271)
- Add support for pushing and pulling private images in the remote builder (#252)
- Parallel obstore should not use TaskGroup for 3.10 (#269)
- [Breaking] Obstore parallel reader (#253)
- Use idl2 in the ray plugin (#266)
- Pin flyteidl to a pre-release (#267)
- Use idl2 proto in the dask plugin (#257)
- fix UvProjectHandler handle function (#264)
- Add PoetryProject and update IDL (#174)
- Just `make fmt` (#263)
- Updates to _deploy logic and bump some actor versions (#262)
- Pin idl2 version (#261)
- Correct syntax of run.wait() in examples (#258)
- Add require_project_and_domain decorator (#256)
- Remove unused connector __init__.py files (#255)
- catch FileNoteFoundError if git is not present (#250)
- [Feat] Enable setting image in cli (#190)
- update resource tuner example (#251)
- Write higher order function that work with plain python udfs  (#249)
- Speed up imports (#247)
- higher order pattern examples (#226)
- Improve error message for type engine (#243)
- Proto to flyteidl2 (#178)
- Ignore dockerignore files when copying files into uv and poetry project (#215)
- Update dataframes example (#238)

## [v2.0.0b24] - 2025-10-02

- build elastic plugin (#237)
- Introduce parent_env_name which is pickable (#236)
- Deployment patterns using current_domain (#235)
- Example configs updated to use git configs (#230)
- Fix fast registration for Dask (#233)
- Don't use --overwrite for macs (#234)

## [v2.0.0b23] - 2025-09-30

- Deploy trigger (#232)
- Improve exception handling in custom excepthook (#227)
- Elastic plugin (#141)
- Bigquery task example (#196)
- **Breaking** flyte.io.File sync/async api and Faster data (#228)
- [FEAT] Enable Local cache (#168)
- PathRewrite system (#222)
- Allow cli to turn off default on flags (#224)
- Adjust layer arg check in remote builder (#225)
- Update plugin examples (#223)
- Fix langgraph example (#221)
- Improved local run experience (#220)
- Improve hash calculation for UV and Poetry project layers (#218)
- [FEAT] Use TaskEnvironment name as image identifier (#211)
- Triggers (alpha - sdk only) (#59)
- Support for Queues (#137)
- Add uv workspace example in deploy_patterns (#208)
- Overwrite files (#213)
- add poetry image builder layer (#176)
- Fix pickle transformer (#206)
- [Breaking] Removes the need for `add_task` in TaskEnvironment (#210)
- Fixed local execution for paths/reports/raw-data etc (#207)
- default interruptible to not be set in task tempalte unless specified (#198)
- Controller should exit and write error message (#197)
- Improve Flyte initialization error messages (#199)
- Allow local builder's command to use mounted secrets, fix secret command generation from env vars (#202)
- Reduce startup overhead for the debug button (#203)
- Disable cache for raw container task by default (#200)
- Allows filtering runs by phase and user (#204)
- This Example demonstrates how to copy the entire source code into an image (#195)
- Scale testing (#189)
- Update Spark example configuration (#193)
- Add an example for submitting rayjob to an existing cluster (#192)
- Update workflow decorator in README example (#194)

## [v2.0.0b22] - 2025-09-16

- [Fix] Image Cache bug when python version different between local and remote (#163)
- Fixed ParamSpec propagation for `@task` decorator (#188)
- Wait for flyte-pypi package availability before building (#184)
- Literals are now returned in native type (#187)
- Stress large runs (#186)
- Small updates (#185)
- Optimize pickle storage based on object size (#179)
- Fix deployment versioning with environment and image cache hash (#180)
- Resolve config only once! (#183)

## [v2.0.0b21] - 2025-09-15

- (ray_example): Increase memory resources for Ray task (#182)
- Handle inputs/outputs and data in notebooks (#181)
- Remove optimize task (#177)
- Build customized CLI apps - pass to Flyte (#175)
- Support Literal type (as enums) and Enums in dataclasses and pydantic models (#170)
- update logger rich styling to work on terminal and jupyter (#171)
- Path.walk is not compatible with Python 3.10 and 3.11 (#146)
- add custom plaintext keyring for google colab (#166)
- remove tutorials that are already in unionai-examples (#164)
- Azure support (#169)
- Runtime should use execution project/domain as the context (#173)
- Not copying ignore group file into docker image (#126)
- add secret file group example (#172)
- Nesting Asyncio (#158)
- Fix docstring indents (#165)
- add graphql example (#159)
- add get git root config to config auto (#162)
- Add example: cached resource tuner (#161)
- update examples to use flyte.git.config_from_root (#156)

## [v2.0.0b20] - 2025-09-08

- Better backoff and logging (#157)
- Controller resiliency (#153)

## [v2.0.0b19] - 2025-09-04

- Spark config can be overriden dynamically (#152)
- Support for functools.partial and flyte.map (#151)
- Update default config path to ./.flyte/config.yaml, add `flyte.git` namespace to provide git utilities (#147)
- Entry log line fix (#150)
- Trace fanout (#136)
- DataFrame cleanup (#145)
- Fixing CLI handling of optional and none types (#148)
- init_from_config supports pathlib.Path (#143)
- Create a new LazyEntity when overriding (#140)
- Inmemory accumulation (#124)
- bugfixes on ml/gridsearch_gpu.py example (#139)
- Reference Tasks are immutable.  (#138)
- Translate pydantic to dataframe (#66)

## [v2.0.0b18] - 2025-08-28

- Fixed issues and tested (#135)
- Trace recovery stressor (#35)
- Trace example add more cases (#133)
- with_uv_project doesnt support extras in uv.lock (#127)
- Trace Outputs can be None, 0 or non existent (#132)
- with_uv_project for remote builder (#129)
- Remove timestamp from the context.tar.gz (#131)
- Use error_info.message for clearer error msgs (#130)
- Updated HashMethod approach (#105)
- Global logging handler (#113)
- remove force=true from build_default_image.py (#122)
- Handle auto image build (#121)
- Find errors early (#120)
- check if a uv sync is up to date (#119)
- Debug button for Ray/Spark (#114)
- Add a default interpreter_path and entrypoint_path (#115)

## [v2.0.0b17] - 2025-08-22

- uv lock update (#117)

## [v2.0.0b16] - 2025-08-22

- Gate IDL to be <2 for now (#116)

## [v2.0.0b15] - 2025-08-22

- Plumb through trace spec interface (#39)

## [v2.0.0b14] - 2025-08-21

- Long running! (#107)
- added uv script to hello_polyglot example (#110)
- Improves log line when authenticating (#111)
- add example for doing gridsearch with gpu ml training (#20)
- Rename `task.friendly_name` -> `task.short_name` May break for some (#106)
- Secret mount for remote builder (#29)
- Debug Button (#101)
- Run distinct directories (#104)
- Only use --prerelease for plugins in CI (#103)

## [v2.0.0b13] - 2025-08-19

- updated lock file b13 (#98)
- Many fanout jobs - 10k (#96)

## [v2.0.0b12] - 2025-08-19

- Remove uv.lock for plugins (#97)
- Fix release (#95)
- Test friendly names (#94)
- Large fanout test (#84)
- Add support PodTemplate override (#80)
- Update is_in_cluster (#82)
- Add Optimize Step trigger after build! (#87)
- Reduce startup penalty by ~400ms (#92)
- Breaking change: Replacing `env` with `env_vars` to disambiguate from an env (#93)
- Fix secret (#91)
- Nix pydantic & dataclass type structure (#90)
- Example: parameterizing image uris for flyte run/deploy via env vars (#88)
- Removes the grpc warning messages. (#85)
- CLI Get for single object always returns json now (#83)
- rename command to flyte run deployed-task (#81)

## [v2.0.0b10] - 2025-08-15

- Add support for running reference tasks (#78)
- added hello_polyglot example (#71)
- Dask Plugin (#73)
- Bypassing storage writes to obstore to using async read/write for metadata when possible (#77)
- Implement Reusable containers autoscaling. (#47)
- Process commands that do not have nouns (#72)
- Remove editable (#76)

## [v2.0.0b9] - 2025-08-14

- Fix requirements template mount (#75)

## [v2.0.0b8] - 2025-08-14

- Update pip package handling in local builder (#74)
- Add support override in LazyEntity (#69)
- Logs now has syncify and handles unavailable logs longer (#70)
- Clean up docstring (#67)
- Click link rendering (#68)
- Wheel changes (#65)
- update plugin readmes (#64)
- Correct tag assignment in dev mode (#63)
- Remove ld library path from local docker builder (#62)

## [v2.0.0b7] - 2025-08-12

- Remove cache behavior "enabled", adjust remote cache return accordingly (#61)
- Add openai plugin: implement drop-in replacement for openai-agents function_tool (#60)
- Example of loading memory (#58)
- Making every command output json (In the future yaml too) (#56)
- Abort & TimedOut propagation (#54)
- Add langgraph example (#41)
- Reduce BlockingIOError (PollerCompletionQueue) log spew with reference tasks and reusable containers (#53)
- improve secrets cli: add validation, provide secret value through input (#51)
- fix example Image arguments (#49)
- AnyIO Example (#48)
- Correct redundant type Union (#45)
- Rusty log lines (#46)
- Optimizing Imports (#42)
- Fancy reports (#40)

## [v2.0.0b6] - 2025-08-06

- Fixes image identifier for cases when certain attributes are relocatable (#38)
- Update agent sim traces (#37)
- Fixed Docs generation and added a test (#34)
- Deployment pattern uvscript (#33)
- You can use pyproject.toml (#32)

## [v2.0.0b5] - 2025-08-05

- Support better image building (#31)
- update agent sim loadtest: add timer (#30)
- Update spark/ray example (#23)
- improve required inputs validation logic (#27)
- Fast repeated runs (#28)
- Recursively deploying Environments. (#17)

## [v2.0.0b4] - 2025-08-04

- Traces are throttled and are run on the background thread (#26)
- Report content type (#25)
- Update README.md (#22)
- Support secret mounts in local image builds (#11)
- Updated Agent sim example and formatting (#19)
- add a simple openai agent example (#7)

## [v2.0.0b3] - 2025-08-02

- Protects Driver memory (#21)

## [v2.0.0b2] - 2025-08-01

- add support for file/dir types in containertask (#18)
- add example of agent simulation load test (#16)
- add ported openai agent example with tools (#15)
- Unpin flyteidl (#14)
- Readme setup (#13)
- Update missing protos (#4)
- Update README.md (#12)
- Fix README (#9)
- update agent examples (#10)
- Remove Migrate from flyte (not yet ready) (#5)
- updated docs (#1)

## [v2.0.0b1] - 2025-07-30

- Add typed interfaces into traces (#2)
- Copy over GH actions (#3)

## [v2.0.0b0] - 2025-07-29

- first commit
