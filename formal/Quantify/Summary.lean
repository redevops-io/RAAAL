/-
  The headline money numbers, and the single authority behind them.

      contributed   the positive flows
      withdrawn     the negative flows, as a magnitude
      final_value   the last portfolio value
      gain          final_value − (contributed − withdrawn)

  Every one of them is a *view* of the same flow series and the same value
  series. That is the property worth proving: a summary recomputed from
  somewhere else can disagree with the ledger it claims to summarise, and the
  disagreement shows up as a headline number nobody can reconcile.

  Integers in minor units, so no Mathlib here — these are sums, not ratios, and
  the lightweight core stays buildable from a bare toolchain.

  Written over `Int` rather than the `Money` abbreviation, and that is not
  cosmetic: `omega` matches on the syntactic type of a local and does not
  unfold an abbreviation, so every arithmetic goal stated in `Money` comes back
  as "no usable constraints". `Ledger.lean` pays the same cost differently, by
  reaching for explicit `Int` lemmas instead.

  **Signs.** A contribution is positive and a withdrawal negative in the flow
  series; `withdrawn` reports the magnitude, because "withdrawn: −500" reads as
  money arriving. The two conventions meet in exactly one place, `netContributed`
  below, which is why that is the theorem that matters.
-/

import Quantify.Types

namespace Quantify
namespace Summary

/-- Money the investor put in, over the whole run. -/
def contributed : List Int → Int
  | []        => 0
  | f :: rest => (if f > 0 then f else 0) + contributed rest

/-- Money the investor took out, as a positive magnitude. -/
def withdrawn : List Int → Int
  | []        => 0
  | f :: rest => (if f < 0 then -f else 0) + withdrawn rest

/-- Every flow, signed. The authority the two halves are views of. -/
def total : List Int → Int
  | []        => 0
  | f :: rest => f + total rest

/-- What the investor is net out of pocket. -/
def netContributed (flows : List Int) : Int :=
  contributed flows - withdrawn flows

/-- What the run made, over and above the money put into it. -/
def gain (flows : List Int) (finalValue : Int) : Int :=
  finalValue - netContributed flows

/-- **Splitting the flows and recombining them recovers the total.**

    The theorem the whole summary rests on. `contributed` and `withdrawn` read
    disjoint halves of one series, and this says nothing is lost or
    double-counted between them — so `netContributed` is the flow series
    itself, not a second opinion about it.

    The step is one case analysis per flow: a positive flow contributes itself
    and withdraws nothing, a negative one withdraws its magnitude and
    contributes nothing, and a zero is in neither half while still being in the
    total.

    What this does *not* catch, tried and recorded rather than assumed: moving
    the boundary from `> 0` to `≥ 0` is invisible here, because a zero adds
    nothing to a sum either way. It would matter to a *count* of contributions
    and does not matter to a total. What the theorem does catch is a flow
    landing in both halves or in neither while non-zero, and a sign convention
    that disagrees with itself between them. -/
theorem net_is_the_flow_total :
    ∀ (flows : List Int), netContributed flows = total flows
  | []        => by simp [netContributed, contributed, withdrawn, total]
  | f :: rest => by
    have ih := net_is_the_flow_total rest
    simp only [netContributed, contributed, withdrawn, total] at *
    by_cases hpos : f > 0
    · by_cases hneg : f < 0
      · omega
      · rw [if_pos hpos, if_neg hneg]; omega
    · by_cases hneg : f < 0
      · rw [if_neg hpos, if_pos hneg]; omega
      · rw [if_neg hpos, if_neg hneg]; omega

/-- **Gain is terminal value less the flow total.** Stated over the authority
    rather than over the two halves, so the headline number cannot be a third
    independent计算. -/
theorem gain_is_value_less_flows (flows : List Int) (finalValue : Int) :
    gain flows finalValue = finalValue - total flows := by
  simp [gain, net_is_the_flow_total flows]

/-! ## A run with money moving both ways

    A fixture with contributions only would let every theorem above hold in the
    accumulation-only world this project spent a long time escaping.
-/

/-- 1000 in, 300 out, 500 in. Net 1200, and the portfolio ended at 1500. -/
def bothDirections : List Int := [100000, -30000, 50000]

#guard contributed bothDirections == 150000
#guard withdrawn bothDirections == 30000
#guard netContributed bothDirections == 120000
#guard total bothDirections == 120000
#guard gain bothDirections 150000 == 30000

/-- Zeros sit in neither half and still count in the total. Kept as a fixture
    for the shape rather than as a discriminator: the boundary between `> 0`
    and `≥ 0` cannot be seen in a sum. -/
def withAZeroFlow : List Int := [100000, 0, -40000]

#guard contributed withAZeroFlow == 100000
#guard withdrawn withAZeroFlow == 40000
#guard netContributed withAZeroFlow == 60000
#guard total withAZeroFlow == 60000

/-- Taking out more than was put in is representable, and the gain still
    reconciles. -/
def moreOutThanIn : List Int := [50000, -80000]

#guard netContributed moreOutThanIn == -30000
#guard gain moreOutThanIn 0 == 30000

end Summary
end Quantify
