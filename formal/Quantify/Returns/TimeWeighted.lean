/-
  Time-weighted return: what the strategy did, not what the depositor did.

  The point of TWR is that money arriving is not performance. A portfolio that
  goes from 100 to 176 after a 50 contribution has not made 76%; it has made
  whatever the manager made, and the contribution is the depositor's, not the
  strategy's. Reporting the first number is the classic way to make a mediocre
  strategy look excellent, and it is arithmetic that nobody would defend once
  it is written down — which is exactly why it is worth writing down.

  Exact rationals throughout. A product of ratios stated over fixed-point
  integers would accumulate a rounding policy into every theorem.
-/

import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.BigOperators.Group.List
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace Quantify
namespace Returns

/-- One valuation interval, between external cash flows.

    The boundaries are where money enters or leaves. Inside a subperiod the
    only thing changing the value is the market, which is the whole reason the
    subperiod is the unit of measurement. -/
structure Subperiod where
  start   : ℚ
  finish  : ℚ
  nonzero : start ≠ 0

/-- What the market did over one interval. -/
def periodReturn (p : Subperiod) : ℚ :=
  (p.finish - p.start) / p.start

/-- The growth factor, which is what composes. -/
def growth (p : Subperiod) : ℚ := 1 + periodReturn p

/-- **Time-weighted return**: chain the growth factors and subtract one. -/
def twr (ps : List Subperiod) : ℚ :=
  (ps.map growth).prod - 1

/-- A subperiod's growth is exactly `finish / start`. The bridge every theorem
    below leans on, and the place a denominator error would live. -/
theorem growth_eq_ratio (p : Subperiod) : growth p = p.finish / p.start := by
  unfold growth periodReturn
  -- Surfaced explicitly. `field_simp` will not go looking for a non-zero
  -- hypothesis inside a structure field, and without it the goal has a
  -- denominator it is not allowed to clear.
  have h : p.start ≠ 0 := p.nonzero
  field_simp

/-- **No periods, no return.** -/
theorem empty_is_zero : twr [] = 0 := by simp [twr]

/-- **One period is just its return.** -/
theorem single_period (p : Subperiod) : twr [p] = periodReturn p := by
  simp [twr, growth]

/-- 100 → 110 → 99: up a tenth, then down a tenth, and the round trip loses a
    hundredth rather than breaking even. -/
theorem up_then_down_loses_one_percent :
    twr [⟨100, 110, by norm_num⟩, ⟨110, 99, by norm_num⟩] = -1 / 100 := by
  norm_num [twr, growth, periodReturn]

/-- **A contribution at a boundary manufactures no performance.**

    Two intervals whose market returns are `r₁` and `r₂`, with `flow` arriving
    between them. The reported figure is `(1+r₁)(1+r₂) - 1` — the flow is
    absent from it entirely, whatever its size.

    This is the theorem worth having. That the formula compiles says nothing;
    that money entering the portfolio cannot move the performance number is the
    property the metric exists for. -/
theorem a_boundary_flow_does_not_change_the_return
    (v₀ r₁ r₂ flow : ℚ) (h₀ : v₀ ≠ 0) (h₁ : v₀ * (1 + r₁) + flow ≠ 0) :
    twr [⟨v₀, v₀ * (1 + r₁), h₀⟩,
         ⟨v₀ * (1 + r₁) + flow, (v₀ * (1 + r₁) + flow) * (1 + r₂), h₁⟩]
      = (1 + r₁) * (1 + r₂) - 1 := by
  simp only [twr, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
             growth_eq_ratio, mul_one]
  rw [mul_div_assoc']
  field_simp
  ring

/-- The contrast, so the theorem above is not mistaken for a triviality. A
    naive "total return" over the same run counts the depositor's money as
    performance: 100 grows to 110, 50 arrives, 160 grows to 176, and the naive
    figure is 26% against a true 21%. -/
theorem the_naive_figure_overstates_it :
    twr [⟨100, 110, by norm_num⟩, ⟨160, 176, by norm_num⟩] = 21 / 100 ∧
    (176 - 100 - 50) / (100 : ℚ) = 26 / 100 := by
  constructor
  · norm_num [twr, growth, periodReturn]
  · norm_num

end Returns
end Quantify
