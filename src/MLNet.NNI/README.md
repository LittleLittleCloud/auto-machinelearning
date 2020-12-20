## Build and Install ##

This repo is a C# console project wrapping a node.js module.

To build the node.js module, run `cd nni_manager && yarn && yarn build`.
Or if you have problems with node.js toolchain, you can simply extract `nni-manager.7z`, which contains node.js runtime and pre-built NNI manager for Windows.

C# part could be built with `dotnet build` and run with `dotnet run`.

As a POC prototype, installation is not yet implemented and you must manually specifiy project path in `HardCode.cs`.

Python interpreter (any version) is required for a temporary hack, no Python library needed.

## Usage ##

`Program.cs` file demonstrates the usage of NNI library.

In short:

```C#
(string, double)[] ParameterAccuracyPairs = await new Nni.Experiment(trialClassName, tunerName, searchSpace).Run(trialNumber)
```

Because NNI run trials in separate processes, there must be an entrance for trial process, which should call `Nni.TrialRuntime.Run(trialClassName)`.
The command to launch trial is specified in `HardCode.cs`.

Trial class should implement `ITrial` interface, which contains only one method: `double Run(string parameter)`.

## Known Issues ##

  + Trial output (including exception info) is redirected to `nni-experiments` folder and cannot be seen on console.
  + If the C# program is interrupted, node.js process might keep running. In this case you need to kill `node.exe` manually.
  + NNI requires a TCP port to operate. It is currently hard-coded to `8080`.

## WIP Items ##

  + Search space interface.
  + Pass trial class as `Type` or generic type parameter instead of name string.
