// <copyright file="BenchmarkTestBase.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MLNet.Benchmark
{
    public abstract class BenchmarkTestBase
    {
        protected void Context_Log(object sender, LoggingEventArgs e)
        {
            if (e.Source == "AutoPipeline")
            {
                Console.WriteLine(e.Message);
            }
        }

        /// <summary>
        /// Get relative path of <paramref name="fileName"/> from TestData Folder.
        /// </summary>
        /// <param name="fileName">file name.</param>
        /// <returns>relative path of <paramref name="fileName"/>.</returns>
        protected string GetFileFromTestData(string fileName)
        {
            var cwd = Environment.CurrentDirectory;
            var testDataDir = new DirectoryInfo("TestData");
            return Path.Combine(testDataDir.FullName, fileName).Replace(cwd, ".");
        }

        /// <summary>
        /// Get relative path of <paramref name="folderName"/> from TestData Folder.
        /// </summary>
        /// <param name="folderName">folder name.</param>
        /// <returns>relative path.</returns>
        protected string GetFolderFromTestData(string folderName)
        {
            var cwd = Environment.CurrentDirectory;
            var testDataDir = new DirectoryInfo("TestData");
            return Path.Combine(testDataDir.FullName, folderName).Replace(cwd, ".");
        }

        public abstract void Run();
    }
}
