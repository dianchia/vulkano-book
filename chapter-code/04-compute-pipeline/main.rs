//! This is the source code of the "Compute pipeline" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::sync::{self, GpuFuture};

fn main() {
    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = {
        let create_info =
            InstanceCreateInfo { flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, ..Default::default() };
        Instance::new(library, create_info).expect("failed to create instance")
    };

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, queue_family_properties)| queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE))
        .expect("couldn't find a compute queue family") as u32;

    let (device, mut queues) = {
        let queue_create_infos = vec![QueueCreateInfo { queue_family_index, ..Default::default() }];
        let create_info = DeviceCreateInfo { queue_create_infos, ..Default::default() };
        Device::new(physical_device, create_info).expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    // Introduction to compute operations

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let data_iter = 0..65536u32;
    let data_buffer = {
        let create_info = BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() };
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        Buffer::from_iter(memory_allocator.clone(), create_info, allocation_info, data_iter)
            .expect("failed to create buffer")
    };

    // Compute pipelines
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
                #version 460

                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                } buf;

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] *= 12;
                }
            "
        }
    }

    let shader = cs::load(device.clone()).expect("failed to create shader module");

    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = {
        let create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap();
        PipelineLayout::new(device.clone(), create_info).expect("failed to create pipeline layout")
    };

    let compute_pipeline = {
        let create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
        ComputePipeline::new(device.clone(), None, create_info).expect("failed to create compute pipeline")
    };

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));

    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();

    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )
    .unwrap();

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let work_group_counts = [1024, 1, 1];
    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .unwrap();
    // Dispatch is marked unsafe because it can cause undefined behavior if not used
    // correctly.
    unsafe {
        command_buffer_builder.dispatch(work_group_counts).unwrap();
    }
    let command_buffer = command_buffer_builder.build().unwrap();

    let future = sync::now(device).then_execute(queue, command_buffer).unwrap().then_signal_fence_and_flush().unwrap();
    future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");
}
