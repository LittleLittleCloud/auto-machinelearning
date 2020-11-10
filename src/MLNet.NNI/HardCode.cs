using System.IO;

namespace MLNet.NNI
{
    // hard-coded for POV
    public class HardCode
    {
        public static string BasePath = Path.GetDirectoryName(typeof(HardCode).Assembly.Location);
        public static string ManagerPath = @"C:\Users\xiaoyuz\source\repos\nni\nni-manager";
        public static string NodePath = Path.Combine(ManagerPath, "node.exe");
        public static string NniManagerPath = Path.Combine(ManagerPath, "dist");
        public static string TrialDir = @"C:\Users\xiaoyuz\source\repos\machinelearning-auto-pipeline\src\MLNet.NNI\";
#if NETCOREAPP3_1
        public static string ExePath = $"dotnet {Path.Combine(BasePath, "nni-lib.dll")}";
#else
        public static string ExePath = $"{Path.Combine(BasePath, "nni-lib.exe")}";
#endif
        public static string TrialCommand = $"{ExePath} --trial ";
        public static string CsPipePath = @"nni-pipe";
        public static string NodePipePath = $"\\\\.\\pipe\\{CsPipePath}";
        public static string OutputFolder = BasePath;
    }
}
