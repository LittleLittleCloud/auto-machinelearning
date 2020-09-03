// <copyright file="Logger.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    internal class Logger : IChannel
    {
        private static Logger instance = new Logger();

        public static Logger Instance => instance;

        public IChannel Channel { get; set; }

        public string ContextDescription => "AutoPipeline";

        public void Dispose()
        {
            this.Channel?.Dispose();
        }

        public void Error(MessageSensitivity sensitivity, string fmt)
        {
            this.Channel?.Error(sensitivity, fmt);
        }

        public void Error(MessageSensitivity sensitivity, string fmt, params object[] args)
        {
            this.Channel?.Error(sensitivity, fmt, args);
        }

        public void Info(MessageSensitivity sensitivity, string fmt)
        {
            this.Channel?.Info(sensitivity, fmt);
        }

        public void Info(MessageSensitivity sensitivity, string fmt, params object[] args)
        {
            this.Channel?.Info(sensitivity, fmt, args);
        }

        public TException Process<TException>(TException ex)
            where TException : Exception
        {
            return this.Channel?.Process(ex);
        }

        public void Send(ChannelMessage msg)
        {
            this.Channel?.Send(msg);
        }

        public void Trace(MessageSensitivity sensitivity, string fmt)
        {
            this.Channel?.Trace(sensitivity, fmt);
        }

        public void Trace(MessageSensitivity sensitivity, string fmt, params object[] args)
        {
            this.Channel?.Trace(sensitivity, fmt, args);
        }

        public void Warning(MessageSensitivity sensitivity, string fmt)
        {
            this.Channel?.Warning(sensitivity, fmt);
        }

        public void Warning(MessageSensitivity sensitivity, string fmt, params object[] args)
        {
            this.Channel?.Warning(sensitivity, fmt, args);
        }

        // for test usage
        internal Logger(IChannel channel)
        {
            this.Channel = channel;
        }

        private Logger()
        {
        }
    }
}
