//! This is the source code of the first three subchapters from the "Using images" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync;
use vulkano::sync::GpuFuture;

pub fn main() {
    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = {
        let create_info =
            InstanceCreateInfo { flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, ..Default::default() };
        Instance::new(library, create_info).expect("failed to create instance")
    };

    let physical = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("couldn't find a graphical queue family") as u32;

    let (device, mut queues) = {
        let queue_create_infos = vec![QueueCreateInfo { queue_family_index, ..Default::default() }];
        let create_info = DeviceCreateInfo { queue_create_infos, ..Default::default() };
        Device::new(physical, create_info).expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let cmd_alloc = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    // Image creation
    let image = {
        let create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        };
        let allocation_info =
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() };
        Image::new(mem_alloc.clone(), create_info, allocation_info).unwrap()
    };

    let buf = {
        let create_info = BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() };
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        };
        let iter = (0..1024 * 1024 * 4).map(|_| 0u8);
        Buffer::from_iter(mem_alloc.clone(), create_info, allocation_info, iter).expect("failed to create buffer")
    };

    let mut builder = AutoCommandBufferBuilder::primary(
        cmd_alloc.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        })
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buf.clone()))
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(device).then_execute(queue, command_buffer).unwrap().then_signal_fence_and_flush().unwrap();

    future.wait(None).unwrap();

    // Exporting the result
    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

    println!("Everything succeeded!");
}
