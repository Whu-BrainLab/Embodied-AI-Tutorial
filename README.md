# 动手做具身智能：从算法到实机部署的“通关”实践指南

作者：Brain实验室

<p align="center">
  <img src=".\img\brain_lab_icon.png" width="1000">
</p>

## 一、机器人基础操作与硬件集成

- 具体教程：
  - 1.1 机器人的基本操作, [link](https://www.unitree.com/)
  - 1.2 末端执行器安装, [link](https://www.unitree.com/)
  - 1.3 感知传感器安装, [link](https://www.unitree.com/)
  - 1.4 内部计算单元配置, [link](https://www.unitree.com/)
  - 1.5 外部计算单元配置, coming soon
- 硬件设备：
  - [Unitree G1](https://www.unitree.com/g1), is a humanoid robot, [website](https://www.unitree.com/) / [specs](https://www.unitree.com/g1) / [development guide](https://support.unitree.com/home/en/G1_developer/about_G1)
  - [BrainCo Revo2](https://www.brainco.cn/#/product/revo2), is a dexterous hand, [website](https://www.brainco.cn/#/) / [specs](https://www.brainco.cn/#/product/revo2)
  - [HBVCAM-4M2214HD-2 V11](https://item.taobao.com/item.htm?app=chrome&bxsign=scdHwhRBuiO7BSlw6TUeA8eXkWiVj0Die7bH5tTz8trnm1Nj3Qqww4um9U6O2ELVdi0SCN2WuhMQi25VCKkhuuzcxPhBPABMCIS8U7ciqFteTGoHeTIwEmzDIxux8ANlt8U&cpp=1&id=680371983853&price=208.8-255.8&shareUniqueId=29145523128&share_crt_v=1&shareurl=true&short_name=h.TaZxgkpfWkNCakg&sourceType=item&sp_tk=S0t6MDNLeXUwNHU%3D&spm=a2159r.13376460.0.0&suid=F62895A1-C091-4A9D-8F05-6112DDF98B0D&tbSocialPopKey=shareItem&un=aca1c85abcb831099169e0c9646dc81f&un_site=0&ut_sk=1.XK4hhVygl%2BUDALDYhb5dM2Kx_21380790_1730982875785.Copy.1&wxsign=tbwKCHTAc0LCX1yHOWAuwK9RU0_RmIvx0SnO6y2Xms1NZ7Estz289sDmK3vi0OOqzxgSZ5fSHtcom-23z6JZ6AizJI0RI4sX4aCC8owQTIykkyvBAJPp9Pixcd3oWxmdu0M), is a USB stereo vision camera module, [specs](https://item.taobao.com/item.htm?app=chrome&bxsign=scdHwhRBuiO7BSlw6TUeA8eXkWiVj0Die7bH5tTz8trnm1Nj3Qqww4um9U6O2ELVdi0SCN2WuhMQi25VCKkhuuzcxPhBPABMCIS8U7ciqFteTGoHeTIwEmzDIxux8ANlt8U&cpp=1&id=680371983853&price=208.8-255.8&shareUniqueId=29145523128&share_crt_v=1&shareurl=true&short_name=h.TaZxgkpfWkNCakg&sourceType=item&sp_tk=S0t6MDNLeXUwNHU%3D&spm=a2159r.13376460.0.0&suid=F62895A1-C091-4A9D-8F05-6112DDF98B0D&tbSocialPopKey=shareItem&un=aca1c85abcb831099169e0c9646dc81f&un_site=0&ut_sk=1.XK4hhVygl%2BUDALDYhb5dM2Kx_21380790_1730982875785.Copy.1&wxsign=tbwKCHTAc0LCX1yHOWAuwK9RU0_RmIvx0SnO6y2Xms1NZ7Estz289sDmK3vi0OOqzxgSZ5fSHtcom-23z6JZ6AizJI0RI4sX4aCC8owQTIykkyvBAJPp9Pixcd3oWxmdu0M)
  - [HBVCAM-F2209HD V11](https://item.taobao.com/item.htm?app=chrome&bxsign=scdxJiyoXyFQJMC9NeN-HD5WXp1Jm4c97gZg1My1oDINhugIoeU8xMl-vi76fZ4cK3_FM8ftPErUjiYJUwPouSffO8r4hYznHE2028CM2o6UjzG4gT2Lw4oTzv5wIN88Iyu&cpp=1&id=679094218803&price=134.8-139.8&shareUniqueId=33176585741&share_crt_v=1&shareurl=true&short_name=h.S2YWUJan6ZP8Wqv&sourceType=item&sp_tk=TXFISzR1dldsTGs%3D&spm=a2159r.13376460.0.0&suid=7047f886-f0f8-4f91-bada-b38d9205fd07&tbSocialPopKey=shareItem&tk=MqHK4uvWlLk&un=aca1c85abcb831099169e0c9646dc81f&un_site=0&ut_sk=1.ZyosJJg2NacDAEq%2FT02kKwjL_21646297_1758269772435.Copy.1&wxsign=tbwWCF1Erp1XVowwoFxAEuwU8PdBTw3mXukBD7LpgTKX4FCMCCwgAPoETG3CytRf5_YJi1TWZB7rNvEaYLeINwkAOnYQIohoDk5jQ2OceNP71ln1vOhyogInG5TT8Aulpcu), is a USB camera module, [specs](https://item.taobao.com/item.htm?app=chrome&bxsign=scdxJiyoXyFQJMC9NeN-HD5WXp1Jm4c97gZg1My1oDINhugIoeU8xMl-vi76fZ4cK3_FM8ftPErUjiYJUwPouSffO8r4hYznHE2028CM2o6UjzG4gT2Lw4oTzv5wIN88Iyu&cpp=1&id=679094218803&price=134.8-139.8&shareUniqueId=33176585741&share_crt_v=1&shareurl=true&short_name=h.S2YWUJan6ZP8Wqv&sourceType=item&sp_tk=TXFISzR1dldsTGs%3D&spm=a2159r.13376460.0.0&suid=7047f886-f0f8-4f91-bada-b38d9205fd07&tbSocialPopKey=shareItem&tk=MqHK4uvWlLk&un=aca1c85abcb831099169e0c9646dc81f&un_site=0&ut_sk=1.ZyosJJg2NacDAEq%2FT02kKwjL_21646297_1758269772435.Copy.1&wxsign=tbwWCF1Erp1XVowwoFxAEuwU8PdBTw3mXukBD7LpgTKX4FCMCCwgAPoETG3CytRf5_YJi1TWZB7rNvEaYLeINwkAOnYQIohoDk5jQ2OceNP71ln1vOhyogInG5TT8Aulpcu)

## 二、机器人全身运动控制

- 具体教程：
  - 2.1 基于Isaac Gym的基础步态训练, [link](https://www.unitree.com/)
  - 2.2 运动数据采集（一）：基于人类演示视频的GVHMR, [link](https://www.unitree.com/)
  - 2.3 运动数据采集（二）：基于动作捕捉的GMR, [link](https://www.unitree.com/)
  - 2.4 运动策略生成（一）：基于ASAP & PBHC, [link](https://www.unitree.com/)
  - 2.5 运动策略生成（二）：基于Beyond MIMIC, [link](https://www.unitree.com/)
  - 2.6 运动策略板载部署：基于RoboJuDo, [link](https://www.unitree.com/)
- 参考文献：
  - [arXiv 2025.10](https://arxiv.org/abs/2510.02252), Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking, [code](https://github.com/YanjieZe/GMR)
  - [arXiv 2025.09](https://arxiv.org/abs/2509.10771), RSL-RL: A Learning Library for Robotics Research, [website](https://pypi.org/project/rsl-rl-lib/) / [code](https://github.com/leggedrobotics/rsl_rl)
  - [arXiv 2025.09](https://arxiv.org/abs/2509.10771), KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control, [website](http://kungfubot2-humanoid.github.io/) / [code](https://github.com/TeleHuman/PBHC)
  - [RSS 2025](https://arxiv.org/abs/2502.01143), ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills, [website](https://agile.human2humanoid.com/) / [code](https://github.com/LeCAR-Lab/ASAP)
  - [SIGGRAPH 2024](https://arxiv.org/abs/2409.06662), World-Grounded Human Motion Recovery via Gravity-View Coordinates, [website](https://zju3dv.github.io/gvhmr/) / [code](https://github.com/zju3dv/GVHMR)
  - [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.html), Expressive Body Capture: 3D Hands, Face, and Body From a Single Image, [website](https://smpl-x.is.tue.mpg.de/) / [code](https://github.com/vchoutas/smplx)
- 软件工具：
  - [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), is a repository for reinforcement learning implementation based on Unitree robots, supporting Unitree Go2, H1, H1_2, and G1, [code](https://github.com/unitreerobotics/unitree_rl_gym)
  - [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse), is a multi-simulator framework for humanoid robot sim-to-real learning, [code](https://github.com/LeCAR-Lab/HumanoidVerse)
- 硬件设备：
  - [Unitree G1](https://www.unitree.com/g1), is a humanoid robot, [website](https://www.unitree.com/) / [specs](https://www.unitree.com/g1) / [development guide](https://support.unitree.com/home/en/G1_developer/about_G1)

## 三、机器人操作数据采集

- 具体教程：
  - 3.1 机器人半身遥操作：基于XR Teleoperate框架, [link](https://www.unitree.com/)
  - 3.2 机器人全身遥操作（一）：基于TWIST框架, coming soon
  - 3.3 机器人全身遥操作（二）：基于TWIST2框架, coming soon
- 参考文献：
  - [arXiv 2024.07](https://arxiv.org/abs/2407.01512), Open-TeleVision: Teleoperation with Immersive Active Visual Feedback, [website](https://robot-tv.github.io/) / [code](https://github.com/OpenTeleVision/TeleVision)
- 软件工具：
  - [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate/), aims to implement teleoperation control of a Unitree humanoid robot using XR devices (such as Apple Vision Pro, PICO 4 Ultra Enterprise, or Meta Quest 3), [code](https://github.com/unitreerobotics/xr_teleoperate/)
- 硬件设备：
  - [INSPIRE-ROBOTS RH56E2](https://en.inspire-robots.com/product/rh56e2), is a dexterous hand integrating 17 tactile sensor in a single hand, which setup allows real-time acquisition of tactile information from various parts, [website](https://en.inspire-robots.com/) / [specs](https://en.inspire-robots.com/product/rh56e2)
  - [Orbbec Gemini 335](https://www.orbbec.com/products/stereo-vision-camera/gemini-335/), is a depth + RGB camera, [website](https://www.orbbec.com/) / [specs](https://www.orbbec.com/products/stereo-vision-camera/gemini-335/)
  - [Apple Vision Pro](https://www.apple.com/apple-vision-pro/), is a XR device for teleoperation control, [website](https://www.apple.com/apple-vision-pro/)
  - [PICO 4 Ultra Enterprise](https://www.picoxr.com/global/products/pico4-ultra-enterprise), is a XR device for teleoperation control, [website](https://www.picoxr.com/global/products/pico4-ultra-enterprise)

## 四、机器人环境感知与建模

- 具体教程：
  - 4.1 感知模型接入：基于YOLO与SAM, coming soon
  - 4.2 基于激光 SLAM 的环境建图, [link](https://www.unitree.com/)
  - 4.3 场景自动化生成与增强, [link](https://www.unitree.com/)
  - 4.4 室外环境下的机器人导航, [link](https://www.unitree.com/)
- 参考文献：
  - [arXiv 2025.05](https://arxiv.org/abs/2505.10755), Procedural Generation of Articulated Simulation-Ready Assets, [website](https://infinigen.org/) / [code](https://github.com/princeton-vl/infinigen)
  - [arXiv 2024.07](https://arxiv.org/abs/2407.10943), GRUtopia: Dream General Robots in a City at Scale, [website](https://internrobotics.github.io/user_guide/internutopia/) / [code](https://github.com/InternRobotics/InternUtopia)
  - [TRO 2022](https://ieeexplore.ieee.org/abstract/document/9697912/), FAST-LIO2: Fast Direct LiDAR-inertial Odometry, [code](https://github.com/hku-mars/FAST_LIO)
- 软件工具：
  - [ROS2](https://github.com/ros2), Robotic Operating System2, [code](https://github.com/ros2/ros2)
  - [Navigation2](https://nav2.org/), is a open-source navigation framework and system for ROS2, [website](https://nav2.org/) / [code](https://github.com/ros-navigation/navigation2)
  - [Autonomous Exploration Development Environment](https://github.com/HongbiaoZ/autonomous_exploration_development_environment), is meant for leveraging system development and robot deployment for ground-based autonomous navigation and exploration, [website](https://www.cmu-exploration.com/development-environment) / [code](https://github.com/HongbiaoZ/autonomous_exploration_development_environment)
  - [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), is a reference application built on NVIDIA Omniverse that enables developers to develop, simulate, and test AI-driven robots in physically-based virtual environments, [website](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
- 硬件设备：
  - [AgileX Robotics SCOUT 2.0](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), is a an unmanned ground vehicle (UGV) designed for robotics research and development,  [website](https://global.agilex.ai/) / [specs](https://global.agilex.ai/products/scout-2-0)
  - [LimX Dynamics TRON1](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), is a multi-modal biped robot,  [website](https://www.limxdynamics.com/en/tron1) / [specs](https://www.limxdynamics.com/en/tron1/spec)
  - [UBTECH Walker Tienkung](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), is a full-size humanoid robot, [website](https://www.ubtrobot.com/en/) / [specs](https://www.ubtrobot.com/en/ai-education/products/walker-tienkung/v)
  - [NVIDIA Jetson Orin NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), is a embedded computing platform for small, low-power autonomous machines,  [website](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
  - [Livox AVIA](https://www.livoxtech.com/cn/avia), is a LiDAR which combines compact and lightweight design, featuring an FOV greater than 70°,  [website](https://www.livoxtech.com/) / [specs](https://www.livoxtech.com/avia/specs)
  - [Unicore UM982](https://www.unicorecomm.com/products/detail/26), is a positioning and heading module designed to perform on-chip RTK positioning and dual-antenna heading, [website](https://www.unicorecomm.com/products/detail/26) / [specs](https://www.unicorecomm.com/products/detail/26)

## 五、机器人视觉语言导航

- 具体教程：
  - 5.1 基于OpenNav的视觉语言导航, [link](https://www.unitree.com/)
  - 5.2 基于NaVILA的视觉语言导航, [link](https://www.unitree.com/)
  - 5.3 基于InternNav的视觉语言导航, [link](https://www.unitree.com/)
- 参考文献：
  - [ICRA 2025](https://ieeexplore.ieee.org/abstract/document/11128671), SpatialBot: Precise Spatial Understanding with Vision Language Models, [code](https://github.com/BAAI-DCAI/SpatialBot)
  - [RSS 2025](https://arxiv.org/abs/2412.04453), NaVILA: Legged Robot Vision-Language-Action Model for Navigation, [website](https://navila-bot.github.io/) / [code](https://github.com/AnjieCheng/NaVILA/) / [model](https://huggingface.co/a8cheng/navila-llama3-8b-8f)
  - [2025.09](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf), InternVLA-N1: An Open Dual-System Vision-Language Navigation Foundation Model with Learned Latent Plans, [website](https://internrobotics.github.io/internvla-n1.github.io/) / [code](https://github.com/InternRobotics/InternNav)
  - [arXiv 2025.05](https://arxiv.org/abs/2505.08712), NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance, [website](https://wzcai99.github.io/navigation-diffusion-policy.github.io)
  - [arXiv 2024.09](https://arxiv.org/pdf/2409.18794), Open-nav: Exploring zero-shot vision-and-language navigation in continuous environment with open-source llms, [website](https://sites.google.com/view/opennav) / [code](https://github.com/YanyuanQiao/Open-Nav)
  - [arXiv 2024.06](https://arxiv.org/abs/2406.09414), Depth Anything V2, [website](https://depth-anything-v2.github.io/) / [code](https://github.com/DepthAnything/Depth-Anything-V2)
  - [CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/html/Zhang_Recognize_Anything_A_Strong_Image_Tagging_Model_CVPRW_2024_paper.html), Recognize Anything: A Strong Image Tagging Model, [website](https://recognize-anything.github.io/) / [code](https://github.com/xinyu1205/recognize-anything)
  - [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Bridging_the_Gap_Between_Learning_in_Discrete_and_Continuous_Environments_CVPR_2022_paper.html), Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation, [code](https://github.com/YicongHong/Discrete-Continuous-VLN)
  - [ECCV 2020](https://arxiv.org/abs/2004.02857), Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments, [website](https://jacobkrantz.github.io/vlnce/) / [code](https://github.com/jacobkrantz/VLN-CE)
- 软件工具：
  - [Habitat-Sim](https://github.com/facebookresearch/habitat-sim), a flexible, high-performance 3D simulator for embodied AI research, [website](https://aihabitat.org/) / [code](https://github.com/facebookresearch/habitat-sim)
  - [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), a modular high-level library to train embodied AI agents across a variety of tasks and environments, [website](https://aihabitat.org/) / [code](https://github.com/facebookresearch/habitat-lab)

## 六、机器人视觉语言操作
- 具体教程：
  - 6.1 Unitree LeRobot框架介绍及配置, [link](https://www.unitree.com/)
  - 6.2 Unitree LeRobot数据格式及转换, [link](https://www.unitree.com/)
  - 6.3 Unitree LeRobot典型模型库介绍, [link](https://www.unitree.com/)
  - 6.4 Unitree LeRobot训练与工程测试, [link](https://www.unitree.com/)
  - 6.5 NVIDIA GR00T模型微调, [link](https://www.unitree.com/)
  - 6.6 NVIDIA GR00T工程实践, [link](https://www.unitree.com/)
- 参考文献：
  - [arXiv 2025.03](https://arxiv.org/abs/2503.14734), : GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, [website](https://developer.nvidia.com/isaac/gr00t) / [code](https://github.com/NVIDIA/Isaac-GR00T)
  - [arXiv 2024.10](https://arxiv.org/abs/2410.24164), : $π_0$, A Vision-Language-Action Flow Model for General Robot Control, [website](https://www.physicalintelligence.company/blog/pi0) / [code](https://github.com/Physical-Intelligence/openpi)
  - [arXiv 2023.04](https://arxiv.org/abs/2304.13705), Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- 软件工具：
  - [LeRobot](https://github.com/huggingface/lerobot), aims to provide models, datasets, and tools for real-world robotics in PyTorch, [website](https://huggingface.co/docs/lerobot) / [code](https://github.com/huggingface/lerobot)
  - [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot), is a modification of the LeRobot open-source training framework, enabling the training and testing of data collected using the dual-arm dexterous hands of Unitree's G1 robot, [code](https://github.com/unitreerobotics/unitree_IL_lerobot)
  - [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate/), aims to implement teleoperation control of a Unitree humanoid robot using XR devices (such as Apple Vision Pro, PICO 4 Ultra Enterprise, or Meta Quest 3), [code](https://github.com/unitreerobotics/xr_teleoperate/)

## 七、机器人人机交互系统

- 具体教程：
  - 7.1 宇树原生交互系统介绍, [link](https://www.unitree.com/)
  - 7.2 外接大模型的集成方法, [link](https://www.unitree.com/)
  - 7.3 本地知识库构建与接入方案, [link](https://www.unitree.com/)
  - 7.4 机器人导游与导览应用概述, [link](https://www.unitree.com/)
- 软件工具：
  - [LangChain](https://github.com/langchain-ai/langchain), is a framework for building agents and LLM-powered applications, [website](https://docs.langchain.com/oss/python/langchain/overview) / [code](https://github.com/langchain-ai/langchain)
- 硬件设备：
  - [Livox Mid 360](https://www.livoxtech.com/mid-360), is a LiDAR for low speed robotics, [website](https://www.livoxtech.com/mid-360) / [specs](https://www.livoxtech.com/mid-360/specs)

## 联系方式

<img src="./img\contact.png" width="600">


