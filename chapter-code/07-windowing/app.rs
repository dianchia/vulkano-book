use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, VulkanError, VulkanLibrary, sync};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct RenderContext {
    window:             Arc<Window>,
    swapchain:          Arc<Swapchain>,
    render_pass:        Arc<RenderPass>,
    framebuffers:       Vec<Arc<Framebuffer>>,
    pipeline:           Arc<GraphicsPipeline>,
    viewport:           Viewport,
    recreate_swapchain: bool,
    prev_frame_end:     Option<Box<dyn GpuFuture>>,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl From<[f32; 2]> for MyVertex {
    fn from(position: [f32; 2]) -> Self { MyVertex { position } }
}

pub struct App {
    instance:      Arc<Instance>,
    device:        Arc<Device>,
    queue:         Arc<Queue>,
    cmd_alloc:     Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    rcx:           Option<RenderContext>,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let required_ext = Surface::required_extensions(event_loop).unwrap();
        let instance = {
            let create_info = InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_ext,
                ..Default::default()
            };
            Instance::new(library, create_info).expect("failed to create instance")
        };

        let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };
        let (device, queue) = choose_device(&instance, device_ext, event_loop);

        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmd_alloc = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

        let vertices: [MyVertex; 3] = [[-0.5, -0.25].into(), [0.0, 0.5].into(), [0.5, -0.25].into()];
        let vertex_buffer = {
            let create_info = BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() };
            let allocation_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };
            Buffer::from_iter(mem_alloc.clone(), create_info, allocation_info, vertices).unwrap()
        };

        let rcx = None;
        Self { instance, device, queue, cmd_alloc, vertex_buffer, rcx }
    }

    fn redraw(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

        let window_size = rcx.window.inner_size();
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        rcx.prev_frame_end.as_mut().unwrap().cleanup_finished();

        if rcx.recreate_swapchain {
            let create_info = SwapchainCreateInfo { image_extent: window_size.into(), ..rcx.swapchain.create_info() };
            let (swapchain, images) = rcx.swapchain.recreate(create_info).expect("failed to recreate swapchain");
            rcx.swapchain = swapchain;
            rcx.framebuffers = get_frambuffers(&images, rcx.render_pass.clone());
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        let (image_idx, suboptimal, acquired_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {}", e),
            };
        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let render_pass_begin_info = RenderPassBeginInfo {
            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

            ..RenderPassBeginInfo::framebuffer(rcx.framebuffers[image_idx as usize].clone())
        };
        let subpass_begin_info = SubpassBeginInfo {
            // The contents of the first (and only) subpass. This can be either
            // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
            // advanced and is not covered here.
            contents: SubpassContents::Inline,
            ..Default::default()
        };

        #[rustfmt::skip]
        builder
            .begin_render_pass(render_pass_begin_info, subpass_begin_info).unwrap()
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect()).unwrap()
            .bind_pipeline_graphics(rcx.pipeline.clone()).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap();
        unsafe {
            builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0).unwrap();
        }
        builder.end_render_pass(Default::default()).unwrap();

        let cmd_buffer = builder.build().unwrap();

        let swapchain_info = SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_idx);
        let future = rcx
            .prev_frame_end
            .take()
            .unwrap()
            .join(acquired_future)
            .then_execute(self.queue.clone(), cmd_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), swapchain_info)
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => rcx.prev_frame_end = Some(future.boxed()),
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.prev_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => panic!("failed to flush future: {:?}", e),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Vulkano Tutorial");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = create_swapchain(self.device.clone(), surface.clone(), window_size);

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .unwrap();

        let framebuffers = get_frambuffers(&images, render_pass.clone());

        let pipeline = {
            let vs = crate::vs::load(self.device.clone()).unwrap().entry_point("main").unwrap();
            let fs = crate::fs::load(self.device.clone()).unwrap().entry_point("main").unwrap();

            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
            let stages = [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];
            let layout = {
                let create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap();
                PipelineLayout::new(self.device.clone(), create_info).unwrap()
            };
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            let create_info = GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            };

            GraphicsPipeline::new(self.device.clone(), None, create_info).unwrap()
        };

        let viewport = Viewport { offset: [0.0, 0.0], extent: window_size.into(), depth_range: 0.0..=1.0 };

        let recreate_swapchain = false;
        let prev_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain,
            prev_frame_end,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => rcx.recreate_swapchain = true,
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

fn choose_device(
    instance: &Arc<Instance>, device_ext: DeviceExtensions, event_loop: &EventLoop<()>,
) -> (Arc<Device>, Arc<Queue>) {
    let (physical, index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_ext))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS) &&
                        p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");

    let (device, mut queues) = {
        let queue_create_infos = vec![QueueCreateInfo { queue_family_index: index, ..Default::default() }];
        let create_info = DeviceCreateInfo { enabled_extensions: device_ext, queue_create_infos, ..Default::default() };
        Device::new(physical, create_info).expect("could not create device")
    };
    let queue = queues.next().unwrap();

    (device, queue)
}

fn create_swapchain(
    device: Arc<Device>, surface: Arc<Surface>, image_extent: impl Into<[u32; 2]>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let surface_cap = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();
    let (image_format, _) = device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0];
    let composite_alpha = surface_cap.supported_composite_alpha.into_iter().next().unwrap();

    let create_info = SwapchainCreateInfo {
        min_image_count: surface_cap.min_image_count.max(2),
        image_format,
        image_extent: image_extent.into(),
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        composite_alpha,
        ..Default::default()
    };
    Swapchain::new(device.clone(), surface.clone(), create_info).expect("could not create swapchain")
}

fn get_frambuffers(images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let create_info = FramebufferCreateInfo { attachments: vec![view], ..Default::default() };
            Framebuffer::new(render_pass.clone(), create_info).unwrap()
        })
        .collect()
}
