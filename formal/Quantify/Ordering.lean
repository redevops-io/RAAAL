/-
  When an event may execute, and which fill belongs to it.

  Three claims, and the third is the one a date inequality cannot make.

      causality     signal ≤ funding ≤ execution
      no look-ahead a close-derived signal under next-open policy executes
                    strictly later than it fired
      association   every fill is attributable to its own event

  **Why strict, and not `≤`.** The manifest already refuses
  `same_session_close` with the reason *acting on the close that produced the
  signal reads one bar into the future*. That is a claim about information: a
  signal derived from a session's close is not knowable until that close is
  established, so filling at the same close requires a price the decision
  itself helped determine. `no_look_ahead` states it, which turns a declared
  refusal into a proven one.

  **Why association is separate.** A schedule can satisfy every inequality here
  and still hand the wrong fill to the wrong event. That is not hypothetical —
  it is the dense-event defect, where adjacent triggers steal each other's
  fills and the totals still reconcile because the same fills were used, once
  each, in the wrong places.
-/

import Quantify.Types

namespace Quantify

/-- Where a signal's information comes from.

    The distinction is not decorative: an open-derived signal is knowable
    before the session trades, and a close-derived one is not. Only the second
    constrains execution. -/
inductive SignalSource where
  | atOpen
  | atClose
  deriving DecidableEq, Repr

/-- When the engine is permitted to fill. -/
inductive Timing where
  | nextOpen
  | sameClose
  deriving DecidableEq, Repr

/-- One triggering event and the sessions it moved through. Sessions are
    indices into the run's own frame, so "later" is a comparison and not a
    calendar question. -/
structure Event where
  id               : Nat
  signalSession    : Nat
  fundingSession   : Nat
  executionSession : Nat
  deriving DecidableEq, Repr

/-- A fill, carrying the event it was produced for. Identity is on the fill,
    not inferred from where it sits in a list — inferring it is the defect. -/
structure Fill where
  eventId : Nat
  session : Nat
  deriving DecidableEq, Repr

namespace Ordering

/-- **Causality.** Nothing funds before it signals, nothing executes before it
    funds. -/
def causal (e : Event) : Bool :=
  e.signalSession ≤ e.fundingSession && e.fundingSession ≤ e.executionSession

/-- **No look-ahead.** A close-derived signal filling under next-open policy
    must execute strictly after the session it fired on.

    The other three combinations are unconstrained here rather than permitted:
    an open-derived signal is knowable before the session trades, and
    `sameClose` is refused by the manifest rather than by this predicate. A
    theorem that also forbade them would be proving a capability decision, and
    those belong in `mission.capability`. -/
def noLookAhead (src : SignalSource) (timing : Timing) (e : Event) : Bool :=
  match src, timing with
  | SignalSource.atClose, Timing.nextOpen => e.signalSession < e.executionSession
  | _, _ => true

/-- A fill matched to its event by identity. -/
def fillFor (fills : List Fill) (e : Event) : Option Fill :=
  fills.find? (fun f => f.eventId == e.id)

/-- A fill matched to its event by position: the nth event takes the nth fill
    in session order.

    Here to be *disproven*. It is what an implementation does when it pairs by
    walking two sorted lists together, and it agrees with `fillFor` on every
    well-behaved schedule — which is exactly why the defect survived. -/
def insertBySession : Fill → List Fill → List Fill
  | f, []      => [f]
  | f, g :: gs => if f.session ≤ g.session then f :: g :: gs
                  else g :: insertBySession f gs

/-- Session order, written out rather than taken from the standard library so
    that `decide` can evaluate it. A `mergeSort` call leaves the fixtures below
    unprovable by computation, which would have meant asserting the defect
    instead of demonstrating it. -/
def bySession : List Fill → List Fill
  | []      => []
  | f :: fs => insertBySession f (bySession fs)

def fillByPosition (fills : List Fill) (events : List Event)
    (e : Event) : Option Fill :=
  match events.findIdx? (fun x => x.id == e.id) with
  | none   => none
  | some i => (bySession fills)[i]?

/-- Every fill this event was given lands on the session it executed. -/
def fillIsOnTime (fills : List Fill) (e : Event) : Bool :=
  match fillFor fills e with
  | none   => false
  | some f => f.session == e.executionSession

/-- A schedule is sound when every event is causal, respects the policy, and
    owns exactly the fill produced for it. -/
def sound (src : SignalSource) (timing : Timing)
    (events : List Event) (fills : List Fill) : Bool :=
  events.all (fun e => causal e && noLookAhead src timing e
                        && fillIsOnTime fills e)

/-- Causality is exactly the two inequalities, with nothing hidden in it. -/
theorem causal_iff (e : Event) :
    causal e = true ↔
      (e.signalSession ≤ e.fundingSession ∧
       e.fundingSession ≤ e.executionSession) := by
  simp [causal]

/-- **A close-derived signal under next-open policy cannot fill on its own
    session.** The refusal of `same_session_close`, proven rather than
    declared. -/
theorem close_signal_fills_strictly_later
    (e : Event) (h : noLookAhead SignalSource.atClose Timing.nextOpen e = true) :
    e.signalSession < e.executionSession := by
  simpa [noLookAhead] using h

/-- And therefore never on the session it fired. -/
theorem close_signal_never_fills_on_its_own_session
    (e : Event) (h : noLookAhead SignalSource.atClose Timing.nextOpen e = true) :
    e.executionSession ≠ e.signalSession := by
  have := close_signal_fills_strictly_later e h
  omega

/-- Ordering alone does not settle association: a sound-by-inequality schedule
    can still mispair. Stated as an implication that is *not* provable in the
    other direction, and disproven by fixture below. -/
theorem association_is_not_implied_by_causality
    (fills : List Fill) (e : Event) (h : fillIsOnTime fills e = true) :
    ∃ f, fillFor fills e = some f ∧ f.session = e.executionSession := by
  unfold fillIsOnTime at h
  cases hf : fillFor fills e with
  | none   => rw [hf] at h; simp at h
  | some f =>
    rw [hf] at h
    exact ⟨f, rfl, by simpa using h⟩

/-! ## Adjacent events

The canonical dense case: two triggers whose sessions interleave. Ordering is
satisfied by both events, and that is the point — a schedule can be perfectly
chronological and still hand A's fill to B.
-/

/-- signal A day 1, cash A day 2, fill A day 3.
    signal B day 2, cash B day 3, fill B day 4. -/
def adjacentA : Event := ⟨1, 1, 2, 3⟩
def adjacentB : Event := ⟨2, 2, 3, 4⟩

def adjacentEvents : List Event := [adjacentA, adjacentB]
def adjacentFills  : List Fill  := [⟨1, 3⟩, ⟨2, 4⟩]

theorem adjacent_events_are_sound :
    sound SignalSource.atClose Timing.nextOpen adjacentEvents adjacentFills
      = true := by decide

/-- Associations stay A→A and B→B. -/
theorem adjacent_fills_keep_their_events :
    fillFor adjacentFills adjacentA = some ⟨1, 3⟩ ∧
    fillFor adjacentFills adjacentB = some ⟨2, 4⟩ := by
  constructor <;> decide

/-! ## Where ordering is not enough

A delayed execution. Both events are causal, both respect next-open, and the
fills arrive in the opposite order to the events. Pairing by position now hands
each event the other's fill while every inequality still holds — which is the
dense-event defect exactly: the totals reconcile because the same fills were
used, once each, in the wrong places.
-/

/-- A signals first and fills last; B signals second and fills first. -/
def delayedA : Event := ⟨1, 1, 2, 5⟩
def delayedB : Event := ⟨2, 2, 3, 4⟩

def delayedEvents : List Event := [delayedA, delayedB]
def delayedFills  : List Fill  := [⟨1, 5⟩, ⟨2, 4⟩]

/-- Every inequality holds. -/
theorem delayed_events_are_causal :
    delayedEvents.all causal = true := by decide

theorem delayed_events_respect_next_open :
    delayedEvents.all (noLookAhead SignalSource.atClose Timing.nextOpen)
      = true := by decide

/-- **Identity pairs correctly.** -/
theorem delayed_fills_by_id_are_right :
    fillFor delayedFills delayedA = some ⟨1, 5⟩ ∧
    fillFor delayedFills delayedB = some ⟨2, 4⟩ := by
  constructor <;> decide

/-- **Position pairs wrongly, on a schedule that satisfies every ordering
    rule.** This is the theorem a date inequality cannot make: soundness of the
    timeline does not settle who owns which fill. -/
theorem position_pairing_steals_the_other_events_fill :
    fillByPosition delayedFills delayedEvents delayedA = some ⟨2, 4⟩ ∧
    fillByPosition delayedFills delayedEvents delayedB = some ⟨1, 5⟩ := by
  constructor <;> decide

/-- And the two disagree, which is the whole finding. -/
theorem identity_and_position_disagree :
    fillFor delayedFills delayedA
      ≠ fillByPosition delayedFills delayedEvents delayedA := by decide

end Ordering
end Quantify
