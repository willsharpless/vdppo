import dataclasses

import jax
import mujoco.mjx.third_party.mujoco_warp as mjwarp
import warp as wp
from mujoco.mjx._src import types
from mujoco import mjx
from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
from mujoco.mjx.warp import ffi

_m = mjwarp.Model(**{f.name: None for f in dataclasses.fields(mjwarp.Model) if f.init})
_d = mjwarp.Data(**{f.name: None for f in dataclasses.fields(mjwarp.Data) if f.init})
_o = mjwarp.Option(**{f.name: None for f in dataclasses.fields(mjwarp.Option) if f.init})
_s = mjwarp.Statistic(**{f.name: None for f in dataclasses.fields(mjwarp.Statistic) if f.init})
_c = mjwarp.Contact(**{f.name: None for f in dataclasses.fields(mjwarp.Contact) if f.init})
_e = mjwarp.Constraint(**{f.name: None for f in dataclasses.fields(mjwarp.Constraint) if f.init})


@ffi.format_args_for_warp
def _com_pos_shim(
    # Model
    nworld: int,
    nbody: int,
    njnt: int,
    body_inertia: wp.array2d(dtype=wp.vec3),
    body_mass: wp.array2d(dtype=float),
    body_parentid: wp.array(dtype=int),
    body_rootid: wp.array(dtype=int),
    body_subtreemass: wp.array2d(dtype=float),
    body_tree: tuple[wp.array(dtype=int), ...],
    jnt_bodyid: wp.array(dtype=int),
    jnt_dofadr: wp.array(dtype=int),
    jnt_type: wp.array(dtype=int),
    # Data
    cdof: wp.array2d(dtype=wp.spatial_vector),
    cinert: wp.array2d(dtype=mjwp_types.vec10),
    subtree_com: wp.array2d(dtype=wp.vec3),
    xanchor: wp.array2d(dtype=wp.vec3),
    xaxis: wp.array2d(dtype=wp.vec3),
    ximat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
):
    _m.stat = _s
    _m.opt = _o
    _d.efc = _e
    _d.contact = _c
    _m.nbody = nbody
    _m.njnt = njnt
    _m.body_inertia = body_inertia
    _m.body_mass = body_mass
    _m.body_parentid = body_parentid
    _m.body_rootid = body_rootid
    _m.body_subtreemass = body_subtreemass
    _m.body_tree = body_tree
    _m.jnt_bodyid = jnt_bodyid
    _m.jnt_dofadr = jnt_dofadr
    _m.jnt_type = jnt_type
    _d.cdof = cdof
    _d.cinert = cinert
    _d.subtree_com = subtree_com
    _d.xanchor = xanchor
    _d.xaxis = xaxis
    _d.ximat = ximat
    _d.xipos = xipos
    _d.xmat = xmat
    _d.nworld = nworld
    mjwarp.com_pos(_m, _d)


def _com_pos_jax_impl(m: types.Model, d: types.Data):
    output_dims = {
        "cdof": d._impl.cdof.shape,
        "cinert": d._impl.cinert.shape,
        "subtree_com": d.subtree_com.shape,
        "xanchor": d.xanchor.shape,
        "xaxis": d.xaxis.shape,
        "ximat": d.ximat.shape,
        "xipos": d.xipos.shape,
        "xmat": d.xmat.shape,
    }
    jf = ffi.jax_callable_variadic_tuple(
        _com_pos_shim,
        num_outputs=8,
        output_dims=output_dims,
        vmap_method=None,
        in_out_argnames={
            "cdof",
            "cinert",
            "subtree_com",
            "xanchor",
            "xaxis",
            "ximat",
            "xipos",
            "xmat",
        },
    )
    out = jf(
        d.qpos.shape[0],
        m.nbody,
        m.njnt,
        m.body_inertia,
        m.body_mass,
        m.body_parentid,
        m.body_rootid,
        m.body_subtreemass,
        m._impl.body_tree,
        m.jnt_bodyid,
        m.jnt_dofadr,
        m.jnt_type,
        d._impl.cdof,
        d._impl.cinert,
        d.subtree_com,
        d.xanchor,
        d.xaxis,
        d.ximat,
        d.xipos,
        d.xmat,
    )
    d = d.tree_replace(
        {
            "_impl.cdof": out[0],
            "_impl.cinert": out[1],
            "subtree_com": out[2],
            "xanchor": out[3],
            "xaxis": out[4],
            "ximat": out[5],
            "xipos": out[6],
            "xmat": out[7],
        }
    )
    return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def com_pos(m: types.Model, d: types.Data):
    return _com_pos_jax_impl(m, d)


@com_pos.def_vmap
@ffi.marshal_custom_vmap
def com_pos_vmap(unused_axis_size, is_batched, m, d):
    d = com_pos(m, d)
    return d, is_batched[1]


def jac_site(
        m: types.Model, d: types.Data, site_id: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute pair of (NV, 3) Jacobians for a site.

    This is equivalent to mj_jacSite in the CPU backend.

    Args:
      m: MJX Model.
      d: MJX Data (must have kinematics computed).
      site_id: Index of the site.

    Returns:
      Tuple of (jacp, jacr):
        - jacp: Translational Jacobian, shape (nv, 3).
        - jacr: Rotational Jacobian, shape (nv, 3).
    """
    return mjx.jac(m, d, d.site_xpos[site_id], m.site_bodyid[site_id])