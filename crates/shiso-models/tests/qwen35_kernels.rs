//! The qwen35 block launches are the family's own, reached through the engine's
//! trait.
//!
//! The engine holds the weights, the geometry and the scratch; which kernels one
//! block dispatches over them is a fact about the checkpoint family, and this is
//! the case that says so. It asks for nothing a device has to answer: the fake
//! backend reports no thread limit, so the tuner returns the reference shapes
//! and the only thing under test is that the family supplies the trait at all.

use shiso_engine::blocks::BlockKernels;
use shiso_engine::gdn::GdnGeometry;
use shiso_engine::shape::GdnShape;
use shiso_models::qwen35::Qwen35Kernels;

#[path = "common/qwen35_fixture.rs"]
mod fixture;

#[test]
fn the_family_supplies_block_kernels() {
    let backend = fixture::FakeBackend::new();
    let geometry = GdnGeometry::new(&GdnShape::new(32, 16, 128, 4)).expect("widths fit");
    let kernels = Qwen35Kernels::choose(backend.as_ref(), &geometry).expect("a shape is chosen");
    let _: &dyn BlockKernels = &kernels;
}
