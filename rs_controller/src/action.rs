use flyteidl2::google::protobuf::Timestamp;
use pyo3::prelude::*;

use flyteidl2::flyteidl::common::ActionIdentifier;
use flyteidl2::flyteidl::workflow::{ActionUpdate, TraceAction};

use flyteidl2::flyteidl::core::{ExecutionError, TypedInterface};
use flyteidl2::flyteidl::task::{OutputReferences, TaskSpec, TraceSpec};
use flyteidl2::flyteidl::workflow::Phase;

use tracing::debug;

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Task = 0,
    Trace = 1,
}

#[pyclass(dict, get_all, set_all)]
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
    pub phase: Option<Phase>,
    pub started: bool,
    pub retries: u32,
    pub client_err: Option<String>, // Changed from PyErr to String for serializability
    pub cache_key: Option<String>,
    pub queue: Option<String>,
    pub trace: Option<TraceAction>,
}

impl Action {
    pub fn get_run_name(&self) -> String {
        match self.action_id.run.clone() {
            Some(run_id) => run_id.name,
            None => String::from("missing run name"),
        }
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
        self.phase = Some(Phase::Aborted);
    }

    pub fn mark_started(&mut self) {
        debug!("Marking action {:?} as started", self.action_id);
        self.started = true;
        // clear self.task in the future to save memory
    }

    pub fn merge_update(&mut self, obj: &ActionUpdate) {
        if let Ok(new_phase) = Phase::try_from(obj.phase) {
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
        self.started = true;
    }

    pub fn new_from_update(parent_action_name: String, obj: ActionUpdate) -> Self {
        let action_id = obj.action_id.unwrap();
        let phase = Phase::try_from(obj.phase).unwrap();
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
        }
    }

    pub fn is_action_terminal(&self) -> bool {
        if let Some(phase) = &self.phase {
            matches!(
                phase,
                Phase::Succeeded | Phase::Failed | Phase::Aborted | Phase::TimedOut
            )
        } else {
            false
        }
    }

    // action here is the submitted action, invoked by the informer's manual submit.
    pub fn merge_from_submit(&mut self, action: &Action) {
        self.run_output_base = action.run_output_base.clone();
        self.inputs_uri = action.inputs_uri.clone();
        self.group = action.group.clone();
        self.friendly_name = action.friendly_name.clone();

        if !self.started {
            self.task = action.task.clone();
        }

        self.cache_key = action.cache_key.clone();
    }
}

#[pymethods]
impl Action {
    #[staticmethod]
    pub fn from_task(
        sub_action_id: ActionIdentifier,
        parent_action_name: String,
        group_data: Option<String>,
        task_spec: TaskSpec,  // document what this error is
        inputs_uri: String,
        run_output_base: String,
        cache_key: Option<String>,
        queue: Option<String>,
    ) -> Self {
        debug!("Creating Action from task for ID {:?}", &sub_action_id);
        Action {
            action_id: sub_action_id,
            parent_action_name,
            action_type: ActionType::Task,
            friendly_name: task_spec
                .task_template
                .as_ref()
                .and_then(|tt| tt.id.as_ref().and_then(|id| Some(id.name.clone()))),
            group: group_data,
            task: Some(task_spec),
            inputs_uri: Some(inputs_uri),
            run_output_base: Some(run_output_base),
            realized_outputs_uri: None,
            err: None,
            phase: Some(Phase::Unspecified),
            started: false,
            retries: 0,
            client_err: None,
            cache_key,
            queue,
            trace: None,
        }
    }

    /// This creates a new action for tracing purposes. It is used to track the execution of a trace
    #[staticmethod]
    pub fn from_trace(
        parent_action_name: String,
        action_id: ActionIdentifier,
        friendly_name: String,
        group_data: Option<String>,
        inputs_uri: String,
        outputs_uri: String,
        start_time: f64, // Unix timestamp in seconds with fractional seconds
        end_time: f64,   // Unix timestamp in seconds with fractional seconds
        run_output_base: String,
        report_uri: Option<String>,
        typed_interface: Option<TypedInterface>,
    ) -> Self {
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
            phase: Phase::Succeeded.into(),
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

        Action {
            action_id,
            parent_action_name,
            action_type: ActionType::Trace,
            friendly_name: Some(friendly_name),
            group: group_data,
            task: None,
            inputs_uri: Some(inputs_uri),
            run_output_base: Some(run_output_base),
            realized_outputs_uri: Some(outputs_uri),
            phase: Phase::Succeeded.into(),
            err: None,
            started: true,
            retries: 0,
            client_err: None,
            cache_key: None,
            queue: None,
            trace: Some(trace_action),
        }
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
}
