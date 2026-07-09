//! End-to-end: a scripted MockModel drives a simulated sermon through the
//! Observer, exactly as the app's async loop will (minus the real model).

use rhema_comprehension::{
    ComprehensionModel, ComprehensionState, Decision, DominantIntent, MockModel, Observer,
    ObserverConfig, OutputSchema, PassageRef, StateTransition,
};

fn state(intent: DominantIntent, passages: &[&str]) -> ComprehensionState {
    ComprehensionState {
        intent,
        supporting_activities: vec![],
        passages: passages.iter().map(|p| PassageRef::new(*p)).collect(),
        topic: None,
        confidence: 0.92,
    }
}

/// One simulated tick: if the observer wants to evaluate, ask the model and
/// apply its decision, collecting any transition.
async fn tick(
    observer: &mut Observer,
    model: &MockModel,
    schema: &OutputSchema,
    now_ms: u64,
    transitions: &mut Vec<StateTransition>,
) {
    if observer.should_evaluate(now_ms, None) {
        let prompt = observer.build_prompt();
        let decision = model.infer(&prompt, schema).await.unwrap();
        if let Some(t) = observer.apply_decision(decision, now_ms) {
            transitions.push(t);
        }
    }
}

#[tokio::test]
async fn simulated_sermon_produces_expected_state_timeline() {
    // The model will, in order: open a story, confirm it, then move to applying.
    let model = MockModel::new(vec![
        Decision::StateChanged { new_state: state(DominantIntent::StoryTelling, &["Luke 15"]) },
        Decision::NoChange,
        Decision::StateChanged { new_state: state(DominantIntent::Applying, &[]) },
    ]);
    let schema = OutputSchema::default();
    let mut observer = Observer::new(ObserverConfig::default(), 60_000, 4);
    let mut transitions: Vec<StateTransition> = Vec::new();

    // t=0..: the pastor begins retelling the prodigal son.
    observer.ingest_segment("a certain man had two sons", 5_000);
    observer.ingest_segment("the younger asked for his inheritance", 40_000);
    tick(&mut observer, &model, &schema, 60_000, &mut transitions).await; // -> STORY_TELLING

    // A second window: still the same story (model says NO_CHANGE).
    observer.ingest_segment("and he wasted it in a far country", 90_000);
    tick(&mut observer, &model, &schema, 120_000, &mut transitions).await; // -> (no change)

    // The pastor pivots to application.
    observer.ingest_segment("so what does this mean for you", 170_000);
    tick(&mut observer, &model, &schema, 180_000, &mut transitions).await; // -> APPLYING

    // Two transitions recorded (the NO_CHANGE tick produced none).
    assert_eq!(transitions.len(), 2);

    assert_eq!(transitions[0].from, None);
    assert_eq!(transitions[0].to.intent, DominantIntent::StoryTelling);
    assert_eq!(transitions[0].to.passages, vec![PassageRef::new("Luke 15")]);

    // The second transition carries the story as its `from` (the closed state).
    assert_eq!(transitions[1].from.as_ref().map(|s| s.intent), Some(DominantIntent::StoryTelling));
    assert_eq!(transitions[1].to.intent, DominantIntent::Applying);

    // Final live state is APPLYING.
    assert_eq!(observer.current().map(|s| s.intent), Some(DominantIntent::Applying));
}
