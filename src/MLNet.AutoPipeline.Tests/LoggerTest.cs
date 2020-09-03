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
            var logger = new Logger((context as IChannelProvider).Start("AutoPipeline_trace"));
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                if (e.Source == "AutoPipeline_trace")
                {
                    res.Add(e.Message);
                }
            };

            logger.Trace(MessageSensitivity.All, "Hello from AutoPipeline");
            logger.Trace(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline_trace, Kind=Trace] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline_trace, Kind=Trace] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_warning_level()
        {
            var context = new MLContext();
            var logger = new Logger((context as IChannelProvider).Start("AutoPipeline_warning"));
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                if (e.Source == "AutoPipeline_warning")
                {
                    res.Add(e.Message);
                }
            };

            logger.Warning(MessageSensitivity.All, "Hello from AutoPipeline");
            logger.Warning(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline_warning, Kind=Warning] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline_warning, Kind=Warning] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_info_level()
        {
            var context = new MLContext();
            var logger = new Logger((context as IChannelProvider).Start("AutoPipeline_info"));
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                if (e.Source == "AutoPipeline_info")
                {
                    res.Add(e.Message);
                }
            };

            logger.Info(MessageSensitivity.All, "Hello from AutoPipeline");
            logger.Info(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(2);
            res[0].Should().Be("[Source=AutoPipeline_info, Kind=Info] Hello from AutoPipeline");
            res[1].Should().Be("[Source=AutoPipeline_info, Kind=Info] Hello from AutoPipeline");
        }

        [Fact]
        public void Logger_should_log_in_error_level()
        {
            var context = new MLContext();
            var logger = new Logger((context as IChannelProvider).Start("AutoPipeline_error"));
            var res = new List<string>();
            context.Log += (object sender, LoggingEventArgs e) =>
            {
                if (e.Source == "AutoPipeline_error")
                {
                    res.Add(e.Message);
                }
            };

            logger.Error(MessageSensitivity.UserData, "Hello from {0}", "AutoPipeline");

            res.Count.Should().Be(1);
            res[0].Should().Be("[Source=AutoPipeline_error, Kind=Error] Hello from AutoPipeline");
        }
    }
}
