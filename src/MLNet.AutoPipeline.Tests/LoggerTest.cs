// <copyright file="LoggerTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using MLNet.AutoPipeline.Test;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.AutoPipeline.Test
{
    public class LoggerTest : TestBase
    {
        public LoggerTest(ITestOutputHelper outputHelper)
            : base(outputHelper)
        {
        }

        [Fact]
        public void Logger_should_log_in_trace_level()
        {
            var context = new MLContext();
            Logger.Instance.Channel = (context as IChannelProvider).Start("AutoPipeline");
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                res.Add(e.Message);
            };

            Logger.Instance.Trace(MessageSensitivity.All, "Hello from AutoPipeline");
            Logger.Instance.Trace(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline, Kind=Trace] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline, Kind=Trace] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_warning_level()
        {
            var context = new MLContext();
            Logger.Instance.Channel = (context as IChannelProvider).Start("AutoPipeline");
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                res.Add(e.Message);
            };

            Logger.Instance.Warning(MessageSensitivity.All, "Hello from AutoPipeline");
            Logger.Instance.Warning(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline, Kind=Warning] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline, Kind=Warning] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_info_level()
        {
            var context = new MLContext();
            Logger.Instance.Channel = (context as IChannelProvider).Start("AutoPipeline");
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                res.Add(e.Message);
            };

            Logger.Instance.Info(MessageSensitivity.All, "Hello from AutoPipeline");
            Logger.Instance.Info(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline, Kind=Info] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline, Kind=Info] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_error_level()
        {
            var context = new MLContext();
            Logger.Instance.Channel = (context as IChannelProvider).Start("AutoPipeline");
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                res.Add(e.Message);
            };

            Logger.Instance.Error(MessageSensitivity.All, "Hello from AutoPipeline");
            Logger.Instance.Error(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline, Kind=Error] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline, Kind=Error] Hello from AutoPipeline");
        }
    }
}
