using System.IO;

namespace MLNet.NNI
{
    // hard-coded for POV
    public class HardCode
    {
        public static string CsPipePath = @"nni-pipe";
        public static string ManagerPath = @"C:\Users\xiaoyuz\source\repos\nni\nni-manager";
        public static string OutputFolder = BasePath;

        public static string BasePath { get => Path.GetDirectoryName(typeof(HardCode).Assembly.Location); }

        public static string NodePath { get => Path.Combine(ManagerPath, "node.exe"); }

        public static string NniManagerPath { get => Path.Combine(ManagerPath, "dist"); }
#if NETCOREAPP3_1
        public static string ExePath { get => $"dotnet \"{Path.Combine(BasePath, "nni-lib.dll")}\"";}
#else
        public static string ExePath { get => $"\"{Path.Combine(BasePath, "nni-lib.exe")}\""; }
#endif
        public static string TrialCommand { get => $"{ExePath} --trial "; }
        public static string NodePipePath { get => $"\\\\.\\pipe\\{CsPipePath}"; }
    }
}
