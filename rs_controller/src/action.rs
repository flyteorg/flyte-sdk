use flyteidl2::{
    flyteidl::{
        common::{ActionIdentifier, ActionPhase, RunIdentifier},
        core::{ExecutionError, Literal, TypedInterface},
        task::{OutputReferences, TaskSpec, TraceSpec},
        workflow::{ActionUpdate, ConditionAction, TraceAction},
    },
    google::protobuf::Timestamp,
};
use prost::Message;
#[cfg(feature = "py")]
use pyo3::prelude::*;
use tracing::debug;

#[cfg_attr(feature = "py", pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Task = 0,
    Trace = 1,
    /// A condition (signal) action: enqueued PAUSED, resolved out-of-band by a
    /// signal, and its value delivered inline on `ActionUpdate.value`.
    Condition = 2,
}

// No `get_all`/`set_all`: the IDL-typed fields (action_id, task, err, phase,
// trace, condition, condition_output) are not Python-convertible; Python reads
// them only through the explicit `*_bytes`/`*_value` getters in the pymethods
// block below, plus the plain-scalar getters there.
#[cfg_attr(feature = "py", pyclass)]
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub action_id: ActionIdentifier,
    pub parent_action_name: String,
    pub action_type: ActionType,
    pub friendly_name: Option<String>,
    pub group: Option<String>,
    pub task: Option<TaskSpec>,
    pub inputs_uri: Option<String>,
    pub run_output_base: Option<String>,
    pub realized_outputs_uri: Option<String>,
    pub err: Option<ExecutionError>,
    pub phase: Option<ActionPhase>,
    pub started: bool,
    pub retries: u32,
    pub client_err: Option<String>, // Changed from PyErr to String for serializability
    pub cache_key: Option<String>,
    pub queue: Option<String>,
    pub trace: Option<TraceAction>,
    pub condition: Option<ConditionAction>,
    /// The value a condition action was signalled with, delivered inline on
    /// `ActionUpdate.value` rather than through object storage.
    pub condition_output: Option<Literal>,
}

impl Action {
    pub fn get_run_name(&self) -> String {
        let run_name = self
            .action_id
            .run
            .as_ref()
            .expect("Action ID missing run")
            .name
            .clone();
        assert!(!run_name.is_empty());
        run_name
    }

    pub fn get_run_identifier(&self) -> RunIdentifier {
        self.action_id
            .run
            .as_ref()
            .expect("Action ID missing run")
            .clone()
    }

    pub fn get_full_name(&self) -> String {
        format!(
            "{}:{}",
            self.action_id
                .run
                .as_ref()
                .expect("Action ID missing run")
                .name,
            self.action_id.name
        )
    }

    pub fn get_action_name(&self) -> String {
        self.action_id.name.clone()
    }

    pub fn set_client_err(&mut self, err: String) {
        debug!(
            "Setting client error on action {:?} to {}",
            self.action_id, err
        );
        self.client_err = Some(err);
    }

    pub fn mark_cancelled(&mut self) {
        debug!("Marking action {:?} as cancelled", self.action_id);
        self.mark_started();
        self.phase = Some(ActionPhase::Aborted);
    }

    pub fn mark_started(&mut self) {
        debug!("Marking action {:?} as started", self.action_id);
        self.started = true;
        // clear self.task in the future to save memory
    }

    pub fn merge_update(&mut self, obj: &ActionUpdate) {
        if let Ok(new_phase) = ActionPhase::try_from(obj.phase) {
            if self.phase.is_none() || self.phase != Some(new_phase) {
                self.phase = Some(new_phase);
                if obj.error.is_some() {
                    self.err = obj.error.clone();
                }
            }
        }
        if !obj.output_uri.is_empty() {
            self.realized_outputs_uri = Some(obj.output_uri.clone());
        }
        // Condition actions carry their resolved value inline. Captured un-gated
        // (rather than only when action_type == Condition) because a cache entry
        // created by `new_from_update` is typed Task until a later submit repairs
        // it -- gating here would drop the value of an already-signalled
        // condition on a retry.
        if obj.value.is_some() {
            self.condition_output = obj.value.clone();
        }
        self.started = true;
    }

    pub fn new_from_update(parent_action_name: String, obj: ActionUpdate) -> Self {
        let action_id = obj.action_id.unwrap();
        let phase = ActionPhase::try_from(obj.phase).unwrap();
        // An ActionUpdate carries no action type, so this entry is typed Task
        // until the matching submit repairs it via `merge_from_submit`.
        Action {
            action_id: action_id.clone(),
            parent_action_name,
            action_type: ActionType::Task,
            friendly_name: None,
            group: None,
            task: None,
            inputs_uri: None,
            run_output_base: None,
            realized_outputs_uri: Some(obj.output_uri),
            err: obj.error,
            phase: Some(phase),
            started: true,
            retries: 0,
            client_err: None,
            cache_key: None,
            queue: None,
            trace: None,
            condition: None,
            condition_output: obj.value,
        }
    }

    pub fn is_action_terminal(&self) -> bool {
        if let Some(phase) = &self.phase {
            matches!(
                phase,
                ActionPhase::Succeeded
                    | ActionPhase::Failed
                    | ActionPhase::Aborted
                    | ActionPhase::TimedOut
                    // Recovered from a prior run: terminal, success-equivalent; output_uri
                    // points at the source run's outputs (consume as-is).
                    | ActionPhase::Recovered
            )
        } else {
            false
        }
    }

    /// True when the action reached a terminal phase that means success.
    ///
    /// `Recovered` counts: the action was adopted as-is from a prior run and did
    /// not execute here, but its outputs are valid and control flow should treat
    /// it exactly like `Succeeded`. Callers should use this rather than testing
    /// `phase == Succeeded`, which silently mistreats a recovered action, or
    /// inverting `Failed`, which also swallows `Aborted` and `TimedOut`.
    pub fn is_action_successful(&self) -> bool {
        matches!(
            self.phase,
            Some(ActionPhase::Succeeded) | Some(ActionPhase::Recovered)
        )
    }

    /// True while a condition action is waiting to be signalled. Not terminal --
    /// a paused action is still in flight, so waiters must keep waiting.
    pub fn is_action_paused(&self) -> bool {
        self.phase == Some(ActionPhase::Paused)
    }

    // action here is the submitted action, invoked by the informer's manual submit.
    pub fn merge_from_submit(&mut self, action: &Action) {
        self.run_output_base = action.run_output_base.clone();
        self.inputs_uri = action.inputs_uri.clone();
        self.group = action.group.clone();
        self.friendly_name = action.friendly_name.clone();

        // A cache entry created from a watch update is typed Task with no spec
        // (see `new_from_update`). The submit is the only thing that knows the
        // real type, so adopt it here -- otherwise a condition whose update
        // arrived first would never be recognised as one, and `launch_task`
        // would find no spec to enqueue.
        self.action_type = action.action_type;
        if action.condition.is_some() {
            self.condition = action.condition.clone();
        }
        if action.trace.is_some() {
            self.trace = action.trace.clone();
        }

        if !self.started {
            self.task = action.task.clone();
        }

        self.cache_key = action.cache_key.clone();
    }

    /// Decode-agnostic construction of a condition action, shared by the
    /// Python-facing `from_condition` factory and the Rust-side tests.
    pub fn build_condition(
        parent_action_name: String,
        action_id: ActionIdentifier,
        condition_action: ConditionAction,
        inputs_uri: String,
        run_output_base: String,
        group_data: Option<String>,
    ) -> Self {
        debug!("Creating Action from condition for ID {:?}", &action_id);
        Action {
            action_id,
            parent_action_name,
            action_type: ActionType::Condition,
            friendly_name: Some(condition_action.name.clone()),
            group: group_data,
            task: None,
            inputs_uri: Some(inputs_uri),
            run_output_base: Some(run_output_base),
            // Conditions deliver their value inline, so there is no outputs URI.
            realized_outputs_uri: None,
            err: None,
            // Unspecified, not Succeeded: unlike a trace (which is recorded after
            // the fact) a condition is launched and resolved later. A terminal
            // phase here would make `is_action_terminal` true immediately and fire
            // completion before any value arrived.
            phase: Some(ActionPhase::Unspecified),
            started: false,
            retries: 0,
            client_err: None,
            cache_key: None,
            queue: None,
            trace: None,
            condition: Some(condition_action),
            condition_output: None,
        }
    }

    /// Construct a task action from serialized wire bytes. Python and Rust use
    /// different generated protobufs, so the bytes-in signature is the
    /// cross-language contract; the Python-facing `from_task` staticmethod
    /// delegates here.
    pub fn from_task(
        sub_action_id_bytes: &[u8],
        parent_action_name: String,
        group_data: Option<String>,
        task_spec_bytes: &[u8],
        inputs_uri: String,
        run_output_base: String,
        cache_key: Option<String>,
        queue: Option<String>,
    ) -> Result<Self, prost::DecodeError> {
        let sub_action_id = ActionIdentifier::decode(sub_action_id_bytes)?;
        let task_spec = TaskSpec::decode(task_spec_bytes)?;

        debug!("Creating Action from task for ID {:?}", &sub_action_id);
        Ok(Action {
            action_id: sub_action_id,
            parent_action_name,
            action_type: ActionType::Task,
            friendly_name: task_spec
                .task_template
                .as_ref()
                .and_then(|tt| tt.id.as_ref().map(|id| id.name.clone())),
            group: group_data,
            task: Some(task_spec),
            inputs_uri: Some(inputs_uri),
            run_output_base: Some(run_output_base),
            realized_outputs_uri: None,
            err: None,
            phase: Some(ActionPhase::Unspecified),
            started: false,
            retries: 0,
            client_err: None,
            cache_key,
            queue,
            trace: None,
            condition: None,
            condition_output: None,
        })
    }

    /// Create a new action for tracing purposes. It is used to track the
    /// execution of a trace.
    pub fn from_trace(
        parent_action_name: String,
        action_id_bytes: &[u8],
        friendly_name: String,
        group_data: Option<String>,
        inputs_uri: String,
        outputs_uri: String,
        start_time: f64, // Unix timestamp in seconds with fractional seconds
        end_time: f64,   // Unix timestamp in seconds with fractional seconds
        run_output_base: String,
        report_uri: Option<String>,
        typed_interface_bytes: Option<&[u8]>,
    ) -> Result<Self, prost::DecodeError> {
        let action_id = ActionIdentifier::decode(action_id_bytes)?;

        let typed_interface = if let Some(bytes) = typed_interface_bytes {
            Some(TypedInterface::decode(bytes)?)
        } else {
            None
        };

        debug!("Creating Action from trace for ID {:?}", &action_id);
        let trace_spec = TraceSpec {
            interface: typed_interface,
        };
        let start_secs = start_time.floor() as i64;
        let start_nanos = ((start_time - start_time.floor()) * 1e9) as i32;

        let end_secs = end_time.floor() as i64;
        let end_nanos = ((end_time - end_time.floor()) * 1e9) as i32;

        // TraceAction expects an optional OutputReferences - let's only include it if there's something to include
        let output_references = if report_uri.is_some() || !outputs_uri.is_empty() {
            Some(OutputReferences {
                output_uri: outputs_uri.clone(),
                report_uri: report_uri.clone().unwrap_or("".to_string()),
            })
        } else {
            None
        };

        let trace_action = TraceAction {
            name: friendly_name.clone(),
            phase: ActionPhase::Succeeded.into(),
            start_time: Some(Timestamp {
                seconds: start_secs,
                nanos: start_nanos,
            }),
            end_time: Some(Timestamp {
                seconds: end_secs,
                nanos: end_nanos,
            }),
            outputs: output_references,
            spec: Some(trace_spec),
        };

        Ok(Action {
            action_id,
            parent_action_name,
            action_type: ActionType::Trace,
            friendly_name: Some(friendly_name),
            group: group_data,
            task: None,
            inputs_uri: Some(inputs_uri),
            run_output_base: Some(run_output_base),
            realized_outputs_uri: Some(outputs_uri),
            phase: ActionPhase::Succeeded.into(),
            err: None,
            started: false,
            retries: 0,
            client_err: None,
            cache_key: None,
            queue: None,
            trace: Some(trace_action),
            condition: None,
            condition_output: None,
        })
    }

    /// Create a condition (signal) action.
    ///
    /// A condition is enqueued like a task but is resolved out-of-band: the
    /// backend inserts it in phase PAUSED, and it becomes terminal only when
    /// someone signals it (or it times out server-side). The signalled value
    /// arrives inline on `ActionUpdate.value` -- never through object storage --
    /// and lands in `condition_output`.
    ///
    /// `condition_action_bytes` is an encoded `ConditionAction`, following the
    /// same convention as `from_trace`'s `typed_interface_bytes`: callers own the
    /// proto and pass it as bytes.
    ///
    /// `inputs_uri` is a placeholder -- conditions have no inputs and nothing is
    /// written there -- but it must be non-empty: the server's enqueue validator
    /// rejects an empty value, and `build_action_scalars` errors on a missing one
    /// before any RPC is made.
    pub fn from_condition(
        parent_action_name: String,
        action_id_bytes: &[u8],
        condition_action_bytes: &[u8],
        inputs_uri: String,
        run_output_base: String,
        group_data: Option<String>,
    ) -> Result<Self, prost::DecodeError> {
        let action_id = ActionIdentifier::decode(action_id_bytes)?;
        let condition_action = ConditionAction::decode(condition_action_bytes)?;
        Ok(Self::build_condition(
            parent_action_name,
            action_id,
            condition_action,
            inputs_uri,
            run_output_base,
            group_data,
        ))
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Action {
    #[staticmethod]
    #[pyo3(name = "from_task")]
    fn py_from_task(
        sub_action_id_bytes: &[u8],
        parent_action_name: String,
        group_data: Option<String>,
        task_spec_bytes: &[u8],
        inputs_uri: String,
        run_output_base: String,
        cache_key: Option<String>,
        queue: Option<String>,
    ) -> PyResult<Self> {
        Self::from_task(
            sub_action_id_bytes,
            parent_action_name,
            group_data,
            task_spec_bytes,
            inputs_uri,
            run_output_base,
            cache_key,
            queue,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to decode action: {e}"))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_trace")]
    fn py_from_trace(
        parent_action_name: String,
        action_id_bytes: &[u8],
        friendly_name: String,
        group_data: Option<String>,
        inputs_uri: String,
        outputs_uri: String,
        start_time: f64,
        end_time: f64,
        run_output_base: String,
        report_uri: Option<String>,
        typed_interface_bytes: Option<&[u8]>,
    ) -> PyResult<Self> {
        Self::from_trace(
            parent_action_name,
            action_id_bytes,
            friendly_name,
            group_data,
            inputs_uri,
            outputs_uri,
            start_time,
            end_time,
            run_output_base,
            report_uri,
            typed_interface_bytes,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to decode action: {e}"))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_condition")]
    fn py_from_condition(
        parent_action_name: String,
        action_id_bytes: &[u8],
        condition_action_bytes: &[u8],
        inputs_uri: String,
        run_output_base: String,
        group_data: Option<String>,
    ) -> PyResult<Self> {
        Self::from_condition(
            parent_action_name,
            action_id_bytes,
            condition_action_bytes,
            inputs_uri,
            run_output_base,
            group_data,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to decode action: {e}"))
        })
    }

    #[getter(run_name)]
    fn run_name(&self) -> String {
        self.get_run_name()
    }

    #[getter(name)]
    fn name(&self) -> String {
        self.get_action_name()
    }

    fn has_error(&self) -> bool {
        self.err.is_some() || self.client_err.is_some()
    }

    /// Terminal and successful, counting `Recovered` (see `is_action_successful`).
    #[pyo3(name = "is_successful")]
    fn py_is_successful(&self) -> bool {
        self.is_action_successful()
    }

    /// Waiting to be signalled (condition actions).
    #[pyo3(name = "is_paused")]
    fn py_is_paused(&self) -> bool {
        self.is_action_paused()
    }

    /// The signalled value of a condition action, as serialized `Literal` bytes.
    #[getter]
    fn condition_output_bytes(&self) -> Option<Vec<u8>> {
        self.condition_output.as_ref().map(|v| v.encode_to_vec())
    }

    /// Get action_id as serialized bytes for Python interop
    #[getter]
    fn action_id_bytes(&self) -> PyResult<Vec<u8>> {
        Ok(self.action_id.encode_to_vec())
    }

    /// Get err as serialized bytes for Python interop (returns None if no error)
    #[getter]
    fn err_bytes(&self) -> Option<Vec<u8>> {
        self.err.as_ref().map(|e| e.encode_to_vec())
    }

    /// Get task as serialized bytes for Python interop (returns None if no task)
    #[getter]
    fn task_bytes(&self) -> Option<Vec<u8>> {
        self.task.as_ref().map(|t| t.encode_to_vec())
    }

    /// Get phase as i32 for Python interop (returns None if no phase)
    #[getter]
    fn phase_value(&self) -> Option<i32> {
        self.phase.map(|p| p as i32)
    }

    // Plain-scalar fields Python reads directly; the IDL-typed fields cross
    // only as serialized bytes (see the `*_bytes` getters above).
    #[getter]
    fn parent_action_name(&self) -> &str {
        &self.parent_action_name
    }

    #[getter]
    fn run_output_base(&self) -> Option<&str> {
        self.run_output_base.as_deref()
    }

    #[getter]
    fn realized_outputs_uri(&self) -> Option<&str> {
        self.realized_outputs_uri.as_deref()
    }

    #[getter]
    fn client_err(&self) -> Option<&str> {
        self.client_err.as_deref()
    }

    #[getter]
    fn friendly_name(&self) -> Option<&str> {
        self.friendly_name.as_deref()
    }

    #[getter]
    fn group(&self) -> Option<&str> {
        self.group.as_deref()
    }

    #[getter]
    fn inputs_uri(&self) -> Option<&str> {
        self.inputs_uri.as_deref()
    }

    #[getter]
    fn cache_key(&self) -> Option<&str> {
        self.cache_key.as_deref()
    }

    #[getter]
    fn queue(&self) -> Option<&str> {
        self.queue.as_deref()
    }

    #[getter]
    fn started(&self) -> bool {
        self.started
    }

    #[getter]
    fn retries(&self) -> u32 {
        self.retries
    }
}

#[cfg(test)]
mod tests {
    use flyteidl2::flyteidl::core::{literal::Value, primitive, Literal, Primitive, Scalar};

    use super::*;

    fn action_id(name: &str) -> ActionIdentifier {
        ActionIdentifier {
            run: Some(RunIdentifier {
                org: "org".into(),
                project: "proj".into(),
                domain: "dev".into(),
                name: "run".into(),
            }),
            name: name.into(),
        }
    }

    fn bool_literal(v: bool) -> Literal {
        Literal {
            value: Some(Value::Scalar(Box::new(Scalar {
                value: Some(scalar_primitive(v)),
            }))),
            ..Default::default()
        }
    }

    fn scalar_primitive(v: bool) -> flyteidl2::flyteidl::core::scalar::Value {
        flyteidl2::flyteidl::core::scalar::Value::Primitive(Primitive {
            value: Some(primitive::Value::Boolean(v)),
        })
    }

    fn update(name: &str, phase: ActionPhase, value: Option<Literal>) -> ActionUpdate {
        ActionUpdate {
            action_id: Some(action_id(name)),
            phase: phase as i32,
            error: None,
            output_uri: String::new(),
            value,
        }
    }

    fn condition(name: &str) -> Action {
        let spec = ConditionAction {
            name: name.into(),
            ..Default::default()
        };
        // Keep the encode/decode round-trip so the wire path stays covered,
        // but go through the pyo3-free constructor.
        Action::build_condition(
            "a0".into(),
            action_id(name),
            ConditionAction::decode(spec.encode_to_vec().as_slice())
                .expect("round-trip ConditionAction"),
            "s3://base/c1/inputs.pb".into(),
            "s3://base".into(),
            None,
        )
    }

    #[test]
    fn from_condition_is_launchable_and_not_pre_resolved() {
        let action = condition("c1");
        assert_eq!(action.action_type, ActionType::Condition);
        assert_eq!(
            action.condition.as_ref().map(|c| c.name.as_str()),
            Some("c1")
        );
        // Must be launchable: started=false and a spec present, or launch_task
        // silently skips the enqueue.
        assert!(!action.started);
        // Not terminal at birth, or completion would fire before any value.
        assert!(!action.is_action_terminal());
        assert_eq!(action.phase, Some(ActionPhase::Unspecified));
        // The value arrives inline, never via an outputs URI.
        assert!(action.realized_outputs_uri.is_none());
        assert!(action.condition_output.is_none());
    }

    #[test]
    fn merge_update_captures_the_signalled_value() {
        let mut action = condition("c1");
        action.merge_update(&update("c1", ActionPhase::Paused, None));
        assert!(action.is_action_paused());
        assert!(!action.is_action_terminal(), "paused is not terminal");
        assert!(action.condition_output.is_none());

        action.merge_update(&update(
            "c1",
            ActionPhase::Succeeded,
            Some(bool_literal(true)),
        ));
        assert!(action.is_action_terminal());
        assert!(action.is_action_successful());
        assert_eq!(action.condition_output, Some(bool_literal(true)));
    }

    #[test]
    fn new_from_update_captures_the_signalled_value() {
        // The watch stream can deliver a condition update before the local submit
        // registers the action; the value must survive that ordering.
        let action = Action::new_from_update(
            "a0".into(),
            update("c1", ActionPhase::Succeeded, Some(bool_literal(false))),
        );
        assert_eq!(action.condition_output, Some(bool_literal(false)));
    }

    #[test]
    fn merge_from_submit_repairs_type_and_spec_after_an_early_update() {
        // Update first: typed Task, no spec, but the value is kept.
        let mut cached = Action::new_from_update(
            "a0".into(),
            update("c1", ActionPhase::Succeeded, Some(bool_literal(true))),
        );
        assert_eq!(cached.action_type, ActionType::Task);
        assert!(cached.condition.is_none());

        // Submit second: adopts the real type and spec without losing the value.
        cached.merge_from_submit(&condition("c1"));
        assert_eq!(cached.action_type, ActionType::Condition);
        assert!(cached.condition.is_some());
        assert_eq!(cached.condition_output, Some(bool_literal(true)));
    }

    #[test]
    fn recovered_is_terminal_and_successful() {
        let mut action = condition("c1");
        action.merge_update(&update(
            "c1",
            ActionPhase::Recovered,
            Some(bool_literal(true)),
        ));
        // Adopted from a prior run: it did not execute here, but its outcome is
        // valid and must be treated exactly like Succeeded.
        assert!(action.is_action_terminal());
        assert!(action.is_action_successful());
        assert_eq!(action.condition_output, Some(bool_literal(true)));
    }

    #[test]
    fn unsuccessful_terminal_phases_are_terminal_but_not_successful() {
        for phase in [
            ActionPhase::Failed,
            ActionPhase::Aborted,
            ActionPhase::TimedOut,
        ] {
            let mut action = condition("c1");
            action.merge_update(&update("c1", phase, None));
            assert!(action.is_action_terminal(), "{phase:?} should be terminal");
            assert!(
                !action.is_action_successful(),
                "{phase:?} should not be successful"
            );
        }
    }

    #[test]
    fn non_terminal_phases_keep_waiters_waiting() {
        for phase in [
            ActionPhase::Unspecified,
            ActionPhase::Queued,
            ActionPhase::WaitingForResources,
            ActionPhase::Initializing,
            ActionPhase::Running,
            ActionPhase::Paused,
        ] {
            let mut action = condition("c1");
            action.merge_update(&update("c1", phase, None));
            assert!(
                !action.is_action_terminal(),
                "{phase:?} should not be terminal"
            );
        }
    }
}
