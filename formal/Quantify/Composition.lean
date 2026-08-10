/-
  The operators, wired together in the order Python wires them.

      prices
        → MA threshold          MovingAverage.lean
        → crossing signal       Triggers.lean
        → contribution
        → next-open execution   Ordering.lean
        → reported ledger       Window.lean, Ledger.lean

  **This proves the wiring, not the operators.** Each module already proves its
  own behaviour, and restating any of it here would be a second copy to drift.
  What is asserted is that the pieces connect: that the threshold feeding the
  trigger is the one the MA produced, that the signal feeding the contribution
  is a crossing and not a state, that the fill is matched by identity, and that
  the reporting boundary is applied to events rather than to data.

  A wiring mutation must break this while every module proof still passes.
  That is the test of whether this file is about composition or is just another
  copy of a local theorem.
-/

import Quantify.Ledger
import Quantify.MovingAverage
import Quantify.Ordering
import Quantify.Triggers
import Quantify.Window

namespace Quantify
namespace Composition

open MA Trigger Window LedgerState

/-- Eleven sessions. Two dips below a 3-session average: one early, one late.

        session   0    1    2    3    4    5    6    7    8    9   10
        price   100  100  100   90  100  100  100  100   85   88  100
-/
def prices : List Int :=
  [100, 100, 100, 90, 100, 100, 100, 100, 85, 88, 100]

/-- The window the average needs, and the period the person asked about. -/
def averageWindow : Nat := 3

/-- The first session a 3-session average exists for. Asserted rather than
    assumed, because every index below is stated relative to it. -/
def firstUsable : Nat := averageWindow - 1

#guard movingAverage averageWindow prices 1 == none
#guard movingAverage averageWindow prices firstUsable == some 100

/-- Observations start where the threshold does. The offset is explicit
    because trigger indices count from the start of *this* list, not from the
    start of the price series — and quietly conflating the two is the shape of
    every off-by-one in this stack. -/
def observations : List Observation :=
  [⟨100,  100⟩,   -- session 2
   ⟨ 90,   96⟩,   -- session 3   below
   ⟨100,   96⟩,   -- session 4
   ⟨100,   96⟩,   -- session 5
   ⟨100,  100⟩,   -- session 6
   ⟨100,  100⟩,   -- session 7
   ⟨ 85,   95⟩,   -- session 8   below
   ⟨ 88,   91⟩,   -- session 9   below
   ⟨100,   91⟩]   -- session 10

-- Every threshold above is the average this series actually produces. The fixture would otherwise be a hand-written table that agrees with nothing.
#guard movingAverage averageWindow prices 3 == some 96
#guard movingAverage averageWindow prices 8 == some 95
#guard movingAverage averageWindow prices 9 == some 91

/-- A trigger index, back in session terms. -/
def sessionOf (i : Nat) : Nat := i + firstUsable

-- Crossings at sessions 3 and 8; the condition also *holds* at 9.
#guard (crossing observations).map sessionOf == [3, 8]
#guard (persistent observations).map sessionOf == [3, 8, 9]

/-- The requested period starts at session 5, so the first crossing is inside
    the frame and outside the report. -/
def reported : Window := ⟨5, 5, 10⟩

/-- **Only the later crossing is reportable.** The early one fed the average
    and moved no money. -/
def reportableSignals : List Nat :=
  reported.reportable ((crossing observations).map sessionOf)

#guard reportableSignals == [8]

/-- The contribution that signal creates, and the fill it earns under
    next-open policy. -/
def contribution : Event := ⟨1, 8, 8, 9⟩
def fill : Fill := ⟨1, 9⟩

#guard Ordering.causal contribution == true
#guard Ordering.noLookAhead SignalSource.atClose Timing.nextOpen contribution == true
#guard Ordering.fillFor [fill] contribution == some ⟨1, 9⟩

/-- **Execution is strictly later than the signal.** -/
theorem execution_is_after_the_signal :
    contribution.signalSession < contribution.executionSession := by decide

/-- The ledger that follows: one contribution of 8800, one fill of a hundred
    shares at 88. -/
def ledger : LedgerState :=
  { openingCash   := 0
    openingShares := fun _ => 0
    contributions := [⟨8800, by decide⟩]
    withdrawals   := []
    buys          := [⟨"VTI", 100 * sharesScale, 88, 8800, by decide⟩]
    sells         := []
    fees          := 0
    feesNonneg    := by decide }

/-- **The whole chain, in one statement.** One reported signal, one
    contribution, one fill, and a ledger that closes. -/
theorem the_chain_reconciles :
    reportableSignals.length = 1 ∧
    ledger.contributed = 8800 ∧
    ledger.endingCash = 0 ∧
    ledger.endingShares "VTI" = 100 * sharesScale := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

/-- **The warm-up crossing contributed nothing.** It is in the frame, because
    the average needed it, and out of the report, because it is not what the
    person asked about. -/
theorem the_early_crossing_moved_no_money :
    reported.inFrame 3 = true ∧
    reported.inReported 3 = false ∧
    3 ∉ reportableSignals := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

/-- **Persistent would have paid twice.** Wiring the trigger to the state
    predicate instead of the transition doubles the reported contributions on
    this series — every module theorem still holding. -/
theorem persistent_wiring_would_pay_twice :
    (reported.reportable ((persistent observations).map sessionOf)).length = 2 := by
  decide

end Composition
end Quantify
