## Welcome!

If you are here, it means you are interested in helping us out. And we want you to know that we really appreciate that. There are many way for you to contribute to this project:

- By offering PR to fix bugs. 
- By giving suggestions on improving APIs
- By giving bug report.

## Getting start!

#### Code Style
Good news, there's no mandidate coding style in this project, as long as your code doesn't maddly increase warning message then it's fine. But keep in mind when there's warning message, don't smuff it by fixing styleCop file -- you should always check your code first.

#### Build & Develop
(I recommend using Visual Studio and develop on Windows. The init and build script are only runnable on a windows system.)

Remember to run `init.cmd` first to install dependencies and restore projects.

To build projects, Open `machinelearning-auto-pipeline.sln` in Visual Studio and build solution. or:

`Build.cmd`

To test projects, use Test Explorer in VS, or:

`Build.cmd -test`

For other usage (like pack), please check refer to `build.ps1`.