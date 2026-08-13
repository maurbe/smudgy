"""High-precision erf() for Taichi, ported from Sun/fdlibm's s_erf.c (the same
rational-approximation algorithm used by glibc, musl, and most libm's).

Accuracy: within ~1 ulp of double precision for |x| < 6 when evaluated in
ti.f64 (i.e. effectively machine-precision, vs. ~1.5e-7 for the 5-term
Abramowitz & Stegun 7.1.26 formula, or ~2.5e-5 for the 3-term 7.1.25 "fast"
formula many codebases use). |x| >= 6 saturates to +-1, which is correct to
double precision anyway (erfc(6) ~ 2e-17).

Note: Taichi has no built-in erf (checked the math module docs/source), and
there's no documented way to call libm's erf() through the FFI from Taichi
kernel code, so this is a from-scratch rational-approximation port rather
than a wrapper around something native.
"""

import taichi as ti

# --- coefficients (fdlibm s_erf.c) ---
_erx = 8.45062911510467529297e-01

_pp0, _pp1, _pp2, _pp3, _pp4 = (
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
)
_qq1, _qq2, _qq3, _qq4, _qq5 = (
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
)

_pa0, _pa1, _pa2, _pa3, _pa4, _pa5, _pa6 = (
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
)
_qa1, _qa2, _qa3, _qa4, _qa5, _qa6 = (
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
)

_ra0, _ra1, _ra2, _ra3, _ra4, _ra5, _ra6, _ra7 = (
    -9.86494403484714822705e-03,
    -6.93858572707181764372e-01,
    -1.05586262253232909814e01,
    -6.23753324503260060396e01,
    -1.62396669462573470355e02,
    -1.84605092906711035994e02,
    -8.12874355063065934246e01,
    -9.81432934416914548592e00,
)
_sa1, _sa2, _sa3, _sa4, _sa5, _sa6, _sa7, _sa8 = (
    1.96512716674392571292e01,
    1.37657754143519042600e02,
    4.34565877475229228821e02,
    6.45387271733267880336e02,
    4.29008140027567833386e02,
    1.08635005541779435134e02,
    6.57024977031928170135e00,
    -6.04244152148580987438e-02,
)

_rb0, _rb1, _rb2, _rb3, _rb4, _rb5, _rb6 = (
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e01,
    -1.60636384855821916062e02,
    -6.37566443368389627722e02,
    -1.02509513161107724954e03,
    -4.83519191608651397019e02,
)
_sb1, _sb2, _sb3, _sb4, _sb5, _sb6, _sb7 = (
    3.03380607434824582924e01,
    3.25792512996573918826e02,
    1.53672958608443695994e03,
    3.19985821950859553908e03,
    2.55305040643316442583e03,
    4.74528541206955367215e02,
    -2.24409524465858183362e01,
)

_INV_035 = 1.0 / 0.35  # ~2.857142857142857


@ti.func
def erf_precise(x: ti.f64) -> ti.f64:
    """erf(x), fdlibm rational-approximation algorithm, ti.f64 in/out."""
    ax = ti.abs(x)
    result = 0.0
    if ax < 0.84375:
        z = x * x
        r = _pp0 + z * (_pp1 + z * (_pp2 + z * (_pp3 + z * _pp4)))
        s = 1.0 + z * (_qq1 + z * (_qq2 + z * (_qq3 + z * (_qq4 + z * _qq5))))
        result = x + x * (r / s)
    elif ax < 1.25:
        s = ax - 1.0
        P = _pa0 + s * (
            _pa1 + s * (_pa2 + s * (_pa3 + s * (_pa4 + s * (_pa5 + s * _pa6))))
        )
        Q = 1.0 + s * (
            _qa1 + s * (_qa2 + s * (_qa3 + s * (_qa4 + s * (_qa5 + s * _qa6))))
        )
        val = _erx + P / Q
        result = val if x >= 0 else -val
    elif ax < 6.0:
        s = 1.0 / (ax * ax)
        R = 0.0
        S = 0.0
        if ax < _INV_035:
            R = _ra0 + s * (
                _ra1
                + s
                * (_ra2 + s * (_ra3 + s * (_ra4 + s * (_ra5 + s * (_ra6 + s * _ra7)))))
            )
            S = 1.0 + s * (
                _sa1
                + s
                * (
                    _sa2
                    + s
                    * (
                        _sa3
                        + s * (_sa4 + s * (_sa5 + s * (_sa6 + s * (_sa7 + s * _sa8))))
                    )
                )
            )
        else:
            R = _rb0 + s * (
                _rb1 + s * (_rb2 + s * (_rb3 + s * (_rb4 + s * (_rb5 + s * _rb6))))
            )
            S = 1.0 + s * (
                _sb1
                + s
                * (_sb2 + s * (_sb3 + s * (_sb4 + s * (_sb5 + s * (_sb6 + s * _sb7)))))
            )
        r = ti.exp(-ax * ax - 0.5625 + R / S) / ax
        result = (1.0 - r) if x >= 0 else (r - 1.0)
    else:
        result = 1.0 if x >= 0 else -1.0
    return result


@ti.func
def erf_precise_f32(x: ti.f32) -> ti.f32:
    """Convenience wrapper: compute internally in f64, return f32.
    Use this if your fields are f32 but you need better-than-1e-7 erf.
    """
    return ti.cast(erf_precise(ti.cast(x, ti.f64)), ti.f32)


# --- sanity check against Python's math.erf ---
if __name__ == "__main__":
    import math

    ti.init(arch=ti.cpu, default_fp=ti.f64)

    @ti.kernel
    def check() -> ti.f64:
        max_err = 0.0
        n = 200000
        for i in range(n):
            x = -8.0 + 16.0 * i / (n - 1)
            e = erf_precise(x)
            # can't call math.erf inside kernel; just return the func result
        return max_err

    xs = [-8.0 + 16.0 * i / 199999 for i in range(200000)]

    @ti.kernel
    def eval_at(x: ti.f64) -> ti.f64:
        return erf_precise(x)

    max_abs_err = 0.0
    for x in xs:
        got = eval_at(x)
        want = math.erf(x)
        max_abs_err = max(max_abs_err, abs(got - want))
    print(f"max abs error over x in [-8, 8]: {max_abs_err:.3e}")
