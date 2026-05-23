<claude-mem-context>
# Memory Context

# [GNN_FL_cf_mMIMO] recent context, 2026-05-22 7:53pm GMT+9

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 48 obs (18,051t read) | 324,334t work | 94% savings

### May 1, 2026
1 1:35p 🔵 Project Memory Exists for GNN_FL_cf_mMIMO
2 " 🔵 Detailed User Profile and Project State Loaded from Memory
3 1:36p 🔵 No Bug Fix Records Found in Previous Session Memory
4 1:40p 🔵 claude-mem CLI Not Available in PATH on wine-4 Host
5 " 🔵 claude-mem v12.4.9 Running via npx on wine-4 Host
S5 User ran /clear to reset the conversation (May 1, 1:40 PM)
### May 14, 2026
S6 Recap of GNN_FL with condensed graph response from server — full architecture review completed (May 14, 10:24 AM)
6 11:10a 🔵 GNN_FL Cell-Free mMIMO Project Architecture Confirmed
7 " 🔵 server_return_GAP: Condensed Graph Response Mechanism Traced
8 " 🔵 Two-Round Client-Server Protocol and UE Feature Augmentation Detail
S7 Research contribution framing for GNN-FL with condensed graph response — four contributions identified and paper abstract drafted (May 14, 11:11 AM)
S27 Rewrite IsacHetNetFL in FedGNN.py to use GAP aggregation (from APHetNetFL_sumrate) with full ISAC graph support including SR nodes and aug_batch['SR'] (May 14, 11:11 AM)
### May 20, 2026
103 8:25p ✅ IsacHetNetFL refactored to use GAP-style server return in FedGNN.py
104 " 🔵 Project file structure and server_return_GAP implementation located
105 " 🔵 IsacHetNetFL has a super() bug and is a verbatim copy of APHetNetFL_sumrate
106 8:26p 🔵 server_return_isac fully mapped: missing aug_batch['SR'] before client_data pack
107 " 🔵 'SR' is a graph node type (Sensing Receiver/Radar target), not sum-rate scalar
108 " 🔵 IsacHetNetFL has no SR node type in dim_dict or convolution layers
110 " 🔵 Models/GNN.py contains SR-aware conv layer with dim_dict['SR'] at line 835
112 " 🔵 IsacHetNet dim_dict uses 'sens_edge' key (not 'sense_edge') — subtle spelling to match in IsacHetNetFL
109 8:27p 🔵 get_global_info sends no SR data to server — SR must come from raw dataLoader in server_return_isac
111 " 🔵 IsacHetNet centralized GNN in GNN.py is the complete reference for SR sensing topology
113 8:28p 🔵 IsacHetNet forward() returns same signature as APHetNetFL_sumrate — IsacHetNetFL can use identical training loop
114 " 🔵 loss_function_isac_sumrate reads 'comm_down' edges and AP.x[2:5] for Fisher sensing components
115 8:33p 🔵 FedGNN.py Missing from Expected Path in GNN_FL_cf_mMIMO Project
116 8:34p 🔵 FedGNN.py Located in Models/ Not Utils/ Directory
117 " 🔵 IsacHetNetFL Has Copy-Paste Bug: Calls Wrong Super().__init__()
118 " 🔵 server_return_GAP Architecture: GAP Node Augmentation for Federated Learning
119 " 🔵 IsacHetNet Model in GNN.py Supports AP/UE/SR Heterogeneous Graph with Sensing Edges
120 8:40p 🟣 IsacHetNetFL Updated with GAP Aggregation in FedGNN.py
121 " 🔵 IsacHetNetFL Has Critical Bug and Missing ISAC Architecture in FedGNN.py
122 " 🔵 server_return_isac Structure and fl_eval_isac_sumrate Still Uses server_return_GAP
123 8:41p 🔵 Complete server_return_isac Internals and aug_batch['SR'] Injection Point Identified
124 " 🔵 ISAC FL Uses comm_down/comm_up Edge Keys While Base FL Uses down/up
125 " 🔵 ISAC Loss Function Uses AP.x Sensing Features Not SR Nodes Directly
126 8:42p 🔵 ISAC Graph Schema and dim_dict Structure Confirmed; IsacHetNetFL Not Yet Instantiated in main_ISAC.py
127 " 🔵 Authoritative ISAC Edge Type Names Confirmed from isac_data.py
128 8:43p 🔵 APConvLayer Constructor Signature and edge_upd Dimension Dependency on src_dim_dict
129 " 🟣 IsacHetNetFL Fully Rewritten with ISAC Graph Support and GAP Aggregation
S93 FL-GNN ISAC performance improvement analysis — identified concrete code-level bottlenecks and ranked improvement strategies (May 20, 8:48 PM)
### May 22, 2026
293 10:19a 🔵 FL-GNN ISAC Project Structure at GNN_FL_cf_mMIMO
294 " 🔵 FL-GNN ISAC Architecture: IsacHetNetFL with Dual Sensing/Comm Graph
295 " 🔵 Project Memory Confirms GNN-FL Architecture and QML Extension Status
296 10:20a 🔵 ISAC Server Aggregation: GAP Node Selection and Bottleneck-Aware Edge Features
S94 SR node federation via augmentation — design suggestion for adding global SR context to sensing branch, no code edits yet (May 22, 10:20 AM)
297 10:27a ⚖️ ISAC Improvement Scope: SR Augmentation Only, No Additional CRLB Loss Term
298 " 🔵 SR Embeddings Already Sent to Server but Never Used for Augmentation
299 " 🔵 UE Augmentation Feature Set: Four Contextual Signals Appended in server_return_isac
S95 User asked to re-show a previously lost code edit suggestion for SR augmentation in a federated GNN training system (May 22, 10:27 AM)
S96 Re-show SR augmentation code edit suggestion for federated GNN ISAC training — confirmed edits are already applied in the codebase (May 22, 10:36 AM)
S109 FedGNN on ISAC improvement suggestions — loss function and aug_graph enhancements without code edits, privacy-preserving federated setting (May 22, 11:03 AM)
300 11:04a 🔵 SR Augmented Encoder Dim Formula in FedGNN.py
301 " 🔵 IsacHetNetFL Instantiation and aug_sr_dim=1 Confirmed in main_ISAC.py
302 " 🔴 SR Augmentation Dimension Mismatch Bug — Missing +1 Scalar Feature
318 4:07p ⚖️ FedGNN on ISAC: Loss Function and Graph Augmentation Improvement Suggestions
319 " 🔵 IsacHetNetFL Architecture and FL Training Pipeline for ISAC FedGNN
326 4:19p 🔵 CRLB is a Graph-Level Scalar, Not Per-Node Like Rate-per-UE
S111 CRLB clarification: it is a graph-level scalar, not per-SR/per-AP — revising aug_graph improvement suggestions accordingly (May 22, 4:20 PM)
**Investigated**: User corrected a modeling assumption: CRLB = (Sigma_a + Sigma_b) - nu*(Sigma_a*Sigma_b - Sigma_c^2) is one scalar per graph realization, not decomposable per SR node or AP node the way per-UE rate is.

**Learned**: - CRLB is a global sensing quality scalar per graph — it cannot be treated as a per-node quantity.
    - Prior Highlight 8 (per-SR-target CRLB as node feature) was architecturally incorrect and needed revision.
    - The correct approach for SR augmentation: broadcast global_crlb scalar to all SR nodes as an appended feature, signaling global sensing regime to the SR conv branch.
    - The CRLB gradient w.r.t. each AP's power (w_a, w_b, w_c) is also a client-level scalar, but is meaningful as a per-AP feature because each AP has distinct geometry (q_a, q_b, q_c) — same gradient, different response per AP.
    - Distinction: CRLB value → broadcast scalar to all nodes; CRLB gradient components (w_a, w_b, w_c) → append to AP nodes as directional sensing pressure signals.

**Completed**: Delivered 10 improvement highlights (5 loss function, 5 aug_graph), then revised Highlights 8 and 9 after user correction:
    - Revised Highlight 8: Broadcast global_crlb scalar to all SR nodes (cat with SR.x and global_sr_context) so the SR branch knows current sensing quality regime.
    - Revised Highlight 9: Append (w_a, w_b, w_c) scalars to each AP node — these are per-client CRLB gradient directions that interact with per-AP geometry (q_a, q_b, q_c), making them useful per-AP features despite being scalar.
    All suggestions remain privacy-preserving (no raw data shared across clients).

**Next Steps**: Session appears to be in Q&A/refinement mode on the improvement suggestions — further clarifications or follow-up questions from the user may continue.


Access 324k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>