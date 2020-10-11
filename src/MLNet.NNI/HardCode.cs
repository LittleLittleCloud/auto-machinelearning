namespace Nni {
    // hard-coded for POV
    class HardCode {
        public static string NodePath = @"C:\Users\xiaoyuz\source\repos\nni\nni-manager\node.exe";
        public static string NniManagerPath = @"C:\Users\xiaoyuz\source\repos\nni\nni-manager\dist";
        public static string TrialDir = @"C:\Users\xiaoyuz\source\repos\machinelearning-auto-pipeline\src\MLNet.NNI\";
        public static string TrialCommand = @"..\..\artifacts\bin\nni-lib\Debug\netcoreapp3.1\nni-lib.exe --trial ";
        public static string CsPipePath = @"nni-pipe";
        public static string NodePipePath = $"\\\\.\\pipe\\{CsPipePath}";
    }
}
