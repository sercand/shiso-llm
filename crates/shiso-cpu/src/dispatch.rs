//! The launch dispatcher: what turns a `Backend::run` call into group-by-group
//! kernel code on host threads.
//!
//! # Argument indexing
//!
//! A launch's `args` hold at most one [`KernelArg::Bytes`] (the by-value
//! struct every kernel here takes) and any number of [`KernelArg::Buffer`]s.
//! A kernel table row's `roles` describe the buffers only, and every `arg`
//! index in this module (in [`Launch::f32`], [`Writes::f32`], an overlap
//! message) is the **buffer ordinal**: 0 is the first `Buffer` in the launch,
//! whatever scalar or struct arguments surround it. The struct is read with
//! [`Launch::bytes`]. A scalar argument is refused, because no kernel in this
//! project passes one and a silently dropped value would compute the wrong
//! thing.
//!
//! # Aliasing: snapshot and staged writes
//!
//! During a launch every buffer is read through an immutable slice of its
//! slab, and a group writes only into a [`Writes`] stage. Once all groups have
//! finished, the stages are committed in group order. That reproduces what a
//! GPU gives a kernel whose input and output views overlap - the input reads
//! its pre-launch value - without splitting slabs, and it lets groups with
//! non-contiguous outputs (a GEMM tile, a paged row) run in parallel. Two
//! *writable* views that overlap are refused: there is no defined winner.
//! A kernel whose groups read what earlier groups wrote declares
//! [`GroupsDisjoint::No`], and runs serially, committing after each group.

#![expect(
    dead_code,
    reason = "the table is empty until the kernel tasks land; each row they add uses another part of this surface"
)]

use std::ops::Range;

use bytemuck::Pod;
use half::f16;
use shiso_core::{
    CoreError, CoreResult, DeviceView, Extent3, FnConst, FnConstValue, KernelArg, KernelRef,
    LaunchArgs,
};

use crate::memory::Slabs;

/// A launch with fewer groups than this runs on the caller's thread: spawning
/// threads costs more than a handful of groups saves. A first guess, re-measured
/// in Task 12 of the CPU backend plan.
pub(crate) const MIN_PARALLEL_GROUPS: u64 = 16;

/// A launch whose grid volume times block volume is below this runs on the
/// caller's thread, for the same reason as [`MIN_PARALLEL_GROUPS`]. A first
/// guess, re-measured in Task 12 of the CPU backend plan.
pub(crate) const MIN_PARALLEL_THREADS: u64 = 8192;

/// How a kernel uses one of its buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Read only.
    In,
    /// Written only.
    Out,
    /// Read, and written where the kernel chooses.
    InOut,
}

/// Whether the groups of a kernel write places no other group reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupsDisjoint {
    /// No group reads what another writes: groups may run in any order, in
    /// parallel.
    Yes,
    /// A group may read what an earlier one wrote: run serially, in order.
    No,
}

/// Which function constants a kernel accepts.
///
/// The shared cases launch with none; the engine sends the constants Metal
/// bakes into a pipeline. Each kernel accepts both spellings, and refuses
/// anything else rather than ignore a constant that would change a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstRule {
    /// No constants.
    None,
    /// Constant 600 (`I16`) absent or equal to `block.y`.
    MulMv,
    /// As [`ConstRule::MulMv`], and 601 (`I16`) absent or one of 4, 8, 16.
    MulMvExt,
    /// 700 and 701 (`Bool`): bounds-check toggles the CPU always honours, so
    /// any value is accepted and ignored.
    MulMm,
}

/// The code of one kernel group: reads through the [`Launch`], writes through
/// the [`Writes`], and is handed its grid coordinates.
pub(crate) type GroupFn = fn(&Launch<'_>, [u32; 3], &mut Writes) -> CoreResult<()>;

/// One row of the kernel table.
#[derive(Clone, Copy)]
pub(crate) struct KernelSpec {
    /// Runs one group.
    pub group: GroupFn,
    /// One role per buffer argument, in buffer order.
    pub roles: &'static [Role],
    /// Which function constants the kernel accepts.
    pub consts: ConstRule,
    /// Whether a block shape is one this kernel's group code can honour.
    pub block: fn(Extent3) -> bool,
    /// Whether groups may run in parallel.
    pub disjoint: GroupsDisjoint,
}

/// The kernel table: `(module, entry, spec)`.
pub(crate) type Table = [(&'static str, &'static str, KernelSpec)];

/// Everything a group may read.
#[derive(Debug)]
pub(crate) struct Launch<'a> {
    /// Number of groups.
    pub grid: Extent3,
    /// Threads per group, which the CPU runs as the group's own loop.
    pub block: Extent3,
    /// The function constants the caller sent, already validated.
    pub consts: &'a [FnConst],
    /// Dynamic shared memory the caller asked for; a CPU group allocates its
    /// own scratch, so this is only informational.
    pub smem_bytes: u32,
    entry: &'static str,
    struct_bytes: Option<&'a [u8]>,
    views: Vec<&'a [u8]>,
}

impl<'a> Launch<'a> {
    /// Decodes the launch's by-value struct argument.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the bytes are not exactly `T`'s size,
    /// and [`CoreError::InvalidValue`] when the launch carries none.
    pub(crate) fn bytes<T: Pod>(&self) -> CoreResult<T> {
        let bytes = self.struct_bytes.ok_or_else(|| CoreError::InvalidValue {
            key: "kernel struct argument",
            value: format!("{}: absent", self.entry),
        })?;
        if bytes.len() != size_of::<T>() {
            return Err(CoreError::ShapeMismatch {
                field: "kernel struct argument bytes",
                expected: size_of::<T>() as u64,
                got: bytes.len() as u64,
            });
        }
        Ok(bytemuck::pod_read_unaligned(bytes))
    }

    /// Buffer `arg` as `f32`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an `arg` the launch lacks, or a view
    /// whose offset or length is not a multiple of four.
    pub(crate) fn f32(&self, arg: usize) -> CoreResult<&'a [f32]> {
        self.cast(arg)
    }

    /// Buffer `arg` as `i32`.
    ///
    /// # Errors
    ///
    /// As [`Launch::f32`].
    pub(crate) fn i32(&self, arg: usize) -> CoreResult<&'a [i32]> {
        self.cast(arg)
    }

    /// Buffer `arg` as `f16`.
    ///
    /// # Errors
    ///
    /// As [`Launch::f32`], for a multiple of two.
    pub(crate) fn f16(&self, arg: usize) -> CoreResult<&'a [f16]> {
        self.cast(arg)
    }

    /// Buffer `arg` as raw bytes.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an `arg` the launch lacks.
    pub(crate) fn raw(&self, arg: usize) -> CoreResult<&'a [u8]> {
        self.views
            .get(arg)
            .copied()
            .ok_or_else(|| CoreError::InvalidValue {
                key: "kernel buffer argument",
                value: format!("{}: no buffer {arg}", self.entry),
            })
    }

    fn cast<T: Pod>(&self, arg: usize) -> CoreResult<&'a [T]> {
        bytemuck::try_cast_slice(self.raw(arg)?).map_err(|source| CoreError::InvalidValue {
            // Refused, not copied: a copy would hide a caller that put a
            // typed view at an offset the device would reject too.
            key: "kernel buffer view",
            value: format!("{}: buffer {arg}: {source}", self.entry),
        })
    }
}

/// One group's staged writes.
#[derive(Debug, Default)]
pub(crate) struct Writes {
    records: Vec<Record>,
}

/// A run of bytes a group asked to write.
#[derive(Debug)]
struct Record {
    arg: usize,
    byte_offset: usize,
    data: Vec<u8>,
}

impl Writes {
    /// Stages `values` at element `at` of buffer `arg`.
    pub(crate) fn f32(&mut self, arg: usize, at: usize, values: &[f32]) {
        self.stage(arg, at, size_of::<f32>(), bytemuck::cast_slice(values));
    }

    /// Stages `values` at element `at` of buffer `arg`.
    pub(crate) fn f16(&mut self, arg: usize, at: usize, values: &[f16]) {
        self.stage(arg, at, size_of::<f16>(), bytemuck::cast_slice(values));
    }

    fn stage(&mut self, arg: usize, at: usize, width: usize, data: &[u8]) {
        self.records.push(Record {
            arg,
            // Saturating, so an absurd index fails the bounds check at commit
            // instead of wrapping into a valid one.
            byte_offset: at.saturating_mul(width),
            data: data.to_vec(),
        });
    }
}

/// A launch that passed every check that needs only the arguments.
struct Plan<'a> {
    spec: &'a KernelSpec,
    entry: &'static str,
    args: &'a LaunchArgs<'a>,
    consts: &'a [FnConst],
    struct_bytes: Option<&'a [u8]>,
    views: Vec<DeviceView<'a>>,
}

impl<'a> Plan<'a> {
    fn new(spec: &'a KernelSpec, kernel: KernelRef, args: &'a LaunchArgs<'a>) -> CoreResult<Self> {
        let entry = kernel.entry;
        check_consts(spec.consts, entry, args.block, kernel.consts)?;
        if !(spec.block)(args.block) {
            return Err(CoreError::InvalidValue {
                key: "kernel block",
                value: format!(
                    "{entry}: {}x{}x{}",
                    args.block.x, args.block.y, args.block.z
                ),
            });
        }
        let mut views = Vec::new();
        let mut struct_bytes = None;
        for arg in args.args {
            match arg {
                KernelArg::Buffer(view) => views.push(*view),
                KernelArg::Bytes(bytes) if struct_bytes.is_none() => struct_bytes = Some(*bytes),
                KernelArg::Bytes(_) => {
                    return Err(CoreError::InvalidValue {
                        key: "kernel struct argument",
                        value: format!("{entry}: more than one"),
                    });
                }
                _ => {
                    return Err(CoreError::InvalidValue {
                        key: "kernel scalar argument",
                        value: format!("{entry}: {arg:?}"),
                    });
                }
            }
        }
        if views.len() != spec.roles.len() {
            return Err(CoreError::ShapeMismatch {
                field: "kernel buffer arguments",
                expected: spec.roles.len() as u64,
                got: views.len() as u64,
            });
        }
        let plan = Self {
            spec,
            entry,
            args,
            consts: kernel.consts,
            struct_bytes,
            views,
        };
        plan.check_outputs_disjoint()?;
        Ok(plan)
    }

    /// Two writable views over the same bytes have no defined winner.
    fn check_outputs_disjoint(&self) -> CoreResult<()> {
        let writable = |i: usize| self.spec.roles[i] != Role::In && !self.views[i].is_empty();
        for i in 0..self.views.len() {
            for j in (i + 1)..self.views.len() {
                if !writable(i) || !writable(j) {
                    continue;
                }
                let (a, b) = (self.views[i], self.views[j]);
                if a.id() == b.id()
                    && a.offset() < b.offset() + b.len()
                    && b.offset() < a.offset() + a.len()
                {
                    return Err(CoreError::InvalidValue {
                        key: "kernel output views",
                        value: format!("{}: args {i} and {j} overlap", self.entry),
                    });
                }
            }
        }
        Ok(())
    }

    /// Snapshots every buffer for reading.
    fn bind<'s>(&'s self, slabs: &'s Slabs) -> CoreResult<Launch<'s>> {
        let views = self
            .views
            .iter()
            .map(|view| slabs.bytes(*view))
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(Launch {
            grid: self.args.grid,
            block: self.args.block,
            consts: self.consts,
            smem_bytes: self.args.smem_bytes,
            entry: self.entry,
            struct_bytes: self.struct_bytes,
            views,
        })
    }

    /// Applies one group's writes.
    fn commit(&self, slabs: &mut Slabs, writes: &Writes) -> CoreResult<()> {
        for record in &writes.records {
            let view = self
                .views
                .get(record.arg)
                .copied()
                .filter(|_| self.spec.roles[record.arg] != Role::In)
                .ok_or_else(|| CoreError::InvalidValue {
                    key: "kernel write",
                    value: format!("{}: buffer {} is not writable", self.entry, record.arg),
                })?;
            let end = record.byte_offset.saturating_add(record.data.len());
            if end as u64 > view.len() {
                return Err(CoreError::OutOfBounds {
                    offset: record.byte_offset as u64,
                    len: record.data.len() as u64,
                    buffer_bytes: view.len(),
                });
            }
            slabs.bytes_mut(view)?[record.byte_offset..end].copy_from_slice(&record.data);
        }
        Ok(())
    }
}

/// Checks the launch's function constants against `rule`.
fn check_consts(
    rule: ConstRule,
    entry: &'static str,
    block: Extent3,
    consts: &[FnConst],
) -> CoreResult<()> {
    let is_block_y = |v: i16| u32::try_from(v).is_ok_and(|v| v == block.y);
    for c in consts {
        let accepted = match (rule, c.index, c.value) {
            (ConstRule::MulMv | ConstRule::MulMvExt, 600, FnConstValue::I16(v)) => is_block_y(v),
            (ConstRule::MulMvExt, 601, FnConstValue::I16(v)) => matches!(v, 4 | 8 | 16),
            (ConstRule::MulMm, 700 | 701, FnConstValue::Bool(_)) => true,
            _ => false,
        };
        if !accepted {
            return Err(CoreError::InvalidValue {
                key: "kernel function constant",
                value: format!("{entry}: {c:?}"),
            });
        }
    }
    Ok(())
}

/// How many threads a launch uses: one unless it is big enough and its groups
/// are independent.
fn workers(spec: &KernelSpec, grid: Extent3, block: Extent3, threads: usize) -> usize {
    let groups = grid.volume();
    let big = groups >= MIN_PARALLEL_GROUPS && groups * block.volume() >= MIN_PARALLEL_THREADS;
    if spec.disjoint == GroupsDisjoint::Yes && big {
        usize::try_from(groups)
            .map_or(threads, |g| threads.min(g))
            .max(1)
    } else {
        1
    }
}

/// Grid coordinates of the `index`th group, x fastest.
fn coords(grid: Extent3, index: u64) -> [u32; 3] {
    let (gx, gy) = (u64::from(grid.x), u64::from(grid.y));
    let at = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    [at(index % gx), at(index / gx % gy), at(index / (gx * gy))]
}

/// Runs the groups in `range` in order, staging each one's writes.
fn run_range(spec: &KernelSpec, launch: &Launch<'_>, range: Range<u64>) -> CoreResult<Vec<Writes>> {
    range
        .map(|g| {
            let mut writes = Writes::default();
            (spec.group)(launch, coords(launch.grid, g), &mut writes)?;
            Ok(writes)
        })
        .collect()
}

/// Runs every group on `workers` threads, returning the stages in group order.
fn run_all(spec: &KernelSpec, launch: &Launch<'_>, workers: usize) -> CoreResult<Vec<Writes>> {
    let groups = launch.grid.volume();
    if workers <= 1 {
        return run_range(spec, launch, 0..groups);
    }
    let per = groups.div_ceil(workers as u64);
    let range = |i: u64| (i * per).min(groups)..((i + 1) * per).min(groups);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (1..workers as u64)
            .map(|i| scope.spawn(move || run_range(spec, launch, range(i))))
            .collect();
        let mut staged = run_range(spec, launch, range(0))?;
        for handle in handles {
            // A panic in kernel code is a bug; surface it on the caller
            // rather than turn it into a missing stage.
            let part = handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            staged.extend(part);
        }
        Ok(staged)
    })
}

/// Runs `kernel` against `slabs`.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for an entry the table lacks;
/// [`CoreError::InvalidValue`] or [`CoreError::ShapeMismatch`] for arguments
/// the kernel does not accept; [`CoreError::OutOfBounds`] for a view or a
/// write that leaves its allocation; whatever the group code reports.
pub(crate) fn execute(
    table: &Table,
    slabs: &mut Slabs,
    kernel: KernelRef,
    args: &LaunchArgs<'_>,
    threads: usize,
) -> CoreResult<()> {
    let spec = table
        .iter()
        .find(|(module, entry, _)| *module == kernel.module && *entry == kernel.entry)
        .map(|(_, _, spec)| spec)
        .ok_or(CoreError::Unsupported {
            what: kernel.entry,
            backend: "cpu",
        })?;
    let plan = Plan::new(spec, kernel, args)?;
    let groups = args.grid.volume();
    match spec.disjoint {
        GroupsDisjoint::No => {
            for g in 0..groups {
                let writes = {
                    let launch = plan.bind(slabs)?;
                    run_range(spec, &launch, g..g + 1)?
                };
                for stage in &writes {
                    plan.commit(slabs, stage)?;
                }
            }
        }
        GroupsDisjoint::Yes => {
            let staged = {
                let launch = plan.bind(slabs)?;
                run_all(spec, &launch, workers(spec, args.grid, args.block, threads))?
            };
            for stage in &staged {
                plan.commit(slabs, stage)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::unnecessary_wraps,
    reason = "fixtures: small indices become floats, the bits are what the tests compare, and group fns must match the table's fallible signature"
)]
mod tests {
    use shiso_core::RawBuffer;

    use super::*;

    const ANY_BLOCK: fn(Extent3) -> bool = |_| true;

    fn scale_group(l: &Launch<'_>, g: [u32; 3], w: &mut Writes) -> CoreResult<()> {
        let src = l.f32(0)?;
        let start = g[0] as usize * 64;
        let end = (start + 64).min(src.len());
        let out: Vec<f32> = src[start..end].iter().map(|v| v * 2.0).collect();
        w.f32(1, start, &out);
        Ok(())
    }

    fn inout_group(l: &Launch<'_>, g: [u32; 3], w: &mut Writes) -> CoreResult<()> {
        let data = l.f32(0)?;
        let start = g[0] as usize * 64;
        let end = (start + 64).min(data.len());
        let out: Vec<f32> = data[start..end].iter().map(|v| v + 1.0).collect();
        w.f32(0, start, &out);
        Ok(())
    }

    fn overrun_group(_: &Launch<'_>, _: [u32; 3], w: &mut Writes) -> CoreResult<()> {
        w.f32(0, 0, &[1.0; 8]);
        Ok(())
    }

    fn noop_group(_: &Launch<'_>, _: [u32; 3], _: &mut Writes) -> CoreResult<()> {
        Ok(())
    }

    /// Group `g` writes `g + 1` to element 0 after reading it: only a serial,
    /// in-order run sees each group's predecessor.
    fn chain_group(l: &Launch<'_>, _: [u32; 3], w: &mut Writes) -> CoreResult<()> {
        w.f32(0, 0, &[l.f32(0)?[0] + 1.0]);
        Ok(())
    }

    const fn spec(
        group: GroupFn,
        roles: &'static [Role],
        consts: ConstRule,
        disjoint: GroupsDisjoint,
    ) -> KernelSpec {
        KernelSpec {
            group,
            roles,
            consts,
            block: ANY_BLOCK,
            disjoint,
        }
    }

    const YES: GroupsDisjoint = GroupsDisjoint::Yes;
    const TABLE: &Table = &[
        (
            "test",
            "kernel_test_scale",
            spec(scale_group, &[Role::In, Role::Out], ConstRule::None, YES),
        ),
        (
            "test",
            "kernel_test_inout",
            spec(inout_group, &[Role::InOut], ConstRule::None, YES),
        ),
        (
            "test",
            "kernel_test_two_out",
            spec(noop_group, &[Role::Out, Role::Out], ConstRule::None, YES),
        ),
        (
            "test",
            "kernel_test_overrun",
            spec(overrun_group, &[Role::Out], ConstRule::None, YES),
        ),
        (
            "test",
            "kernel_test_chain",
            spec(
                chain_group,
                &[Role::InOut],
                ConstRule::None,
                GroupsDisjoint::No,
            ),
        ),
        (
            "test",
            "kernel_test_mv",
            spec(noop_group, &[], ConstRule::MulMv, YES),
        ),
        (
            "test",
            "kernel_test_mv_ext",
            spec(noop_group, &[], ConstRule::MulMvExt, YES),
        ),
        (
            "test",
            "kernel_test_mm",
            spec(noop_group, &[], ConstRule::MulMm, YES),
        ),
        (
            "test",
            "kernel_test_block",
            KernelSpec {
                group: noop_group,
                roles: &[],
                consts: ConstRule::None,
                block: |b| b.x == 32,
                disjoint: YES,
            },
        ),
    ];

    fn kernel(entry: &'static str) -> KernelRef {
        KernelRef::new("test", entry)
    }

    fn floats(slabs: &Slabs, raw: RawBuffer) -> Vec<f32> {
        let view = DeviceView::from_raw(raw, 0, raw.bytes());
        bytemuck::cast_slice(slabs.bytes(view).expect("live")).to_vec()
    }

    fn fill(slabs: &mut Slabs, values: &[f32]) -> RawBuffer {
        let bytes = (values.len() * 4) as u64;
        let raw = slabs.alloc(bytes).expect("alloc");
        let view = DeviceView::from_raw(raw, 0, bytes);
        slabs
            .bytes_mut(view)
            .expect("live")
            .copy_from_slice(bytemuck::cast_slice(values));
        raw
    }

    fn launch_scale(slabs: &mut Slabs, n: usize, threads: usize) -> Vec<f32> {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.37 - 5.0).collect();
        let src = fill(slabs, &input);
        let dst = slabs.alloc((n * 4) as u64).expect("alloc");
        let bytes = (n * 4) as u64;
        let args = [
            KernelArg::Buffer(DeviceView::from_raw(src, 0, bytes)),
            KernelArg::Buffer(DeviceView::from_raw(dst, 0, bytes)),
        ];
        let launch = LaunchArgs::new(
            Extent3::linear(n.div_ceil(64) as u32),
            Extent3::linear(256),
            &args,
        );
        execute(TABLE, slabs, kernel("kernel_test_scale"), &launch, threads).expect("launch");
        floats(slabs, dst)[..n].to_vec()
    }

    #[test]
    fn threads_one_and_many_agree_bit_for_bit() {
        let mut slabs = Slabs::new();
        let n = 4097;
        let spec = &TABLE[0].2;
        assert!(workers(spec, Extent3::linear(65), Extent3::linear(256), 7) > 1);
        let one = launch_scale(&mut slabs, n, 1);
        let many = launch_scale(&mut slabs, n, 7);
        assert_eq!(one.len(), n);
        assert!(
            one.iter()
                .zip(&many)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(
            one[4096].to_bits(),
            ((4096.0f32 * 0.37 - 5.0) * 2.0).to_bits()
        );
    }

    #[test]
    fn small_launches_stay_on_the_caller_thread() {
        let spec = &TABLE[0].2;
        assert_eq!(
            workers(spec, Extent3::linear(15), Extent3::linear(1024), 8),
            1
        );
        assert_eq!(workers(spec, Extent3::linear(16), Extent3::linear(1), 8), 1);
        assert_eq!(
            workers(&TABLE[4].2, Extent3::linear(64), Extent3::linear(256), 8),
            1
        );
    }

    #[test]
    fn an_input_aliasing_an_output_reads_the_pre_launch_value() {
        let mut slabs = Slabs::new();
        let input: Vec<f32> = (0..200).map(|i| i as f32 + 1.0).collect();
        let raw = fill(&mut slabs, &input);
        let view = DeviceView::from_raw(raw, 0, 800);
        let args = [KernelArg::Buffer(view), KernelArg::Buffer(view)];
        let launch = LaunchArgs::new(Extent3::linear(4), Extent3::linear(256), &args);
        execute(TABLE, &mut slabs, kernel("kernel_test_scale"), &launch, 4).expect("launch");
        let got = floats(&slabs, raw);
        for (g, want) in got.iter().zip(&input) {
            assert_eq!(*g, want * 2.0);
        }
    }

    #[test]
    fn inout_reads_its_own_pre_launch_value() {
        let mut slabs = Slabs::new();
        let raw = fill(&mut slabs, &[1.0, 2.0, 3.0, 4.0]);
        let args = [KernelArg::Buffer(DeviceView::from_raw(raw, 0, 16))];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(256), &args);
        execute(TABLE, &mut slabs, kernel("kernel_test_inout"), &launch, 1).expect("launch");
        assert_eq!(floats(&slabs, raw), [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn serial_kernels_see_earlier_groups_writes() {
        let mut slabs = Slabs::new();
        let raw = fill(&mut slabs, &[0.0; 4]);
        let args = [KernelArg::Buffer(DeviceView::from_raw(raw, 0, 16))];
        let launch = LaunchArgs::new(Extent3::linear(5), Extent3::linear(1), &args);
        execute(TABLE, &mut slabs, kernel("kernel_test_chain"), &launch, 4).expect("launch");
        assert_eq!(floats(&slabs, raw)[0], 5.0);
    }

    #[test]
    fn overlapping_outputs_are_refused() {
        let mut slabs = Slabs::new();
        let raw = slabs.alloc(64).expect("alloc");
        let args = [
            KernelArg::Buffer(DeviceView::from_raw(raw, 0, 16)),
            KernelArg::Buffer(DeviceView::from_raw(raw, 12, 16)),
        ];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_two_out"), &launch, 1)
            .expect_err("overlap");
        assert!(
            matches!(
                err,
                CoreError::InvalidValue {
                    key: "kernel output views",
                    ..
                }
            ),
            "{err:?}"
        );
        // Touching but not overlapping is fine.
        let args = [
            KernelArg::Buffer(DeviceView::from_raw(raw, 0, 16)),
            KernelArg::Buffer(DeviceView::from_raw(raw, 16, 16)),
        ];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        execute(TABLE, &mut slabs, kernel("kernel_test_two_out"), &launch, 1).expect("adjacent");
    }

    #[test]
    fn a_misaligned_f32_view_is_refused() {
        let mut slabs = Slabs::new();
        let raw = slabs.alloc(256).expect("alloc");
        let args = [
            KernelArg::Buffer(DeviceView::from_raw(raw, 2, 64)),
            KernelArg::Buffer(DeviceView::from_raw(raw, 128, 64)),
        ];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_scale"), &launch, 1)
            .expect_err("misaligned");
        assert!(
            matches!(
                err,
                CoreError::InvalidValue {
                    key: "kernel buffer view",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_write_past_the_view_is_out_of_bounds() {
        let mut slabs = Slabs::new();
        let raw = slabs.alloc(64).expect("alloc");
        let args = [KernelArg::Buffer(DeviceView::from_raw(raw, 0, 16))];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_overrun"), &launch, 1)
            .expect_err("8 floats into 16 bytes");
        assert!(
            matches!(
                err,
                CoreError::OutOfBounds {
                    len: 32,
                    buffer_bytes: 16,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_view_past_its_slab_is_out_of_bounds() {
        let mut slabs = Slabs::new();
        let raw = slabs.alloc(16).expect("alloc");
        let args = [KernelArg::Buffer(DeviceView::from_raw(raw, 8, 64))];
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_inout"), &launch, 1)
            .expect_err("past the slab");
        assert!(matches!(err, CoreError::OutOfBounds { .. }), "{err:?}");
    }

    #[test]
    fn the_buffer_count_and_block_are_checked() {
        let mut slabs = Slabs::new();
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &[]);
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_scale"), &launch, 1)
            .expect_err("no buffers");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    expected: 2,
                    got: 0,
                    ..
                }
            ),
            "{err:?}"
        );
        let err = execute(TABLE, &mut slabs, kernel("kernel_test_block"), &launch, 1)
            .expect_err("block.x is 1");
        assert!(
            matches!(
                err,
                CoreError::InvalidValue {
                    key: "kernel block",
                    ..
                }
            ),
            "{err:?}"
        );
        let ok = LaunchArgs::new(Extent3::linear(1), Extent3::linear(32), &[]);
        execute(TABLE, &mut slabs, kernel("kernel_test_block"), &ok, 1).expect("block.x is 32");
    }

    #[test]
    fn the_struct_argument_is_decoded_exactly() {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        struct Pair {
            a: u32,
            b: f32,
        }
        let pair = Pair { a: 7, b: 1.5 };
        let bytes = [0u8; 2]
            .iter()
            .chain(bytemuck::bytes_of(&pair))
            .copied()
            .collect::<Vec<u8>>();
        let slabs = Slabs::new();
        let args = [KernelArg::Bytes(&bytes[2..])];
        let launch_args = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &args);
        let table: &Table = &[("test", "k", spec(noop_group, &[], ConstRule::None, YES))];
        let plan = Plan::new(&table[0].2, kernel("k"), &launch_args).expect("plan");
        let launch = plan.bind(&slabs).expect("bind");
        let got: Pair = launch.bytes().expect("decode unaligned bytes");
        assert_eq!((got.a, got.b), (7, 1.5));
        let err = launch
            .bytes::<[u32; 3]>()
            .map(|_| ())
            .expect_err("size differs");
        assert!(
            matches!(
                err,
                CoreError::ShapeMismatch {
                    expected: 12,
                    got: 8,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// `KernelRef` wants `'static` constants; a test leaks a few bytes for them.
    fn leaked(consts: &[FnConst]) -> &'static [FnConst] {
        Box::leak(consts.to_vec().into_boxed_slice())
    }

    fn run_with_consts(entry: &'static str, block_y: u32, consts: &[FnConst]) -> CoreResult<()> {
        let mut slabs = Slabs::new();
        let launch = LaunchArgs::new(Extent3::linear(1), Extent3::new(32, block_y, 1), &[]);
        let kernel = KernelRef::specialized("test", entry, leaked(consts));
        execute(TABLE, &mut slabs, kernel, &launch, 1)
    }

    const fn c(index: u16, value: FnConstValue) -> FnConst {
        FnConst { index, value }
    }

    #[test]
    fn unexpected_function_constants_are_refused() {
        let refused = |r: CoreResult<()>| {
            assert!(
                matches!(
                    r,
                    Err(CoreError::InvalidValue {
                        key: "kernel function constant",
                        ..
                    })
                ),
                "{r:?}"
            );
        };
        // None: any constant is refused.
        refused(run_with_consts(
            "kernel_test_scale",
            1,
            &[c(600, FnConstValue::I16(1))],
        ));
        // MulMv: 600 must equal block.y, be I16, and nothing else is allowed.
        run_with_consts("kernel_test_mv", 2, &[]).expect("absent");
        run_with_consts("kernel_test_mv", 2, &[c(600, FnConstValue::I16(2))]).expect("equal");
        refused(run_with_consts(
            "kernel_test_mv",
            2,
            &[c(600, FnConstValue::I16(600))],
        ));
        refused(run_with_consts(
            "kernel_test_mv",
            2,
            &[c(600, FnConstValue::U16(2))],
        ));
        refused(run_with_consts(
            "kernel_test_mv",
            2,
            &[c(601, FnConstValue::I16(4))],
        ));
        // MulMvExt: 601 is 4, 8 or 16.
        run_with_consts(
            "kernel_test_mv_ext",
            2,
            &[c(600, FnConstValue::I16(2)), c(601, FnConstValue::I16(8))],
        )
        .expect("valid pair");
        refused(run_with_consts(
            "kernel_test_mv_ext",
            2,
            &[c(601, FnConstValue::I16(5))],
        ));
        refused(run_with_consts(
            "kernel_test_mv_ext",
            2,
            &[c(600, FnConstValue::I16(4))],
        ));
        // MulMm: 700 and 701 are Bool toggles; anything else is refused.
        run_with_consts(
            "kernel_test_mm",
            1,
            &[
                c(700, FnConstValue::Bool(true)),
                c(701, FnConstValue::Bool(false)),
            ],
        )
        .expect("toggles");
        refused(run_with_consts(
            "kernel_test_mm",
            1,
            &[c(700, FnConstValue::U32(1))],
        ));
        refused(run_with_consts(
            "kernel_test_mm",
            1,
            &[c(702, FnConstValue::Bool(true))],
        ));
    }
}
