// <copyright file="IValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;

namespace MLNet.Sweeper
{
    /// <summary>
    /// This is the interface that each type of parameter sweep needs to implement.
    /// </summary>
    public interface IValueGenerator
    {
        /// <summary>
        /// Given a value in the [0,1] range, return a value for this parameter.
        /// </summary>
        IParameterValue CreateFromNormalized(double normalizedValue);

        /// <summary>
        /// create <see cref="IParameterValue"/> from value text.
        /// </summary>
        /// <param name="valueText">value in text format.</param>
        /// <returns><see cref="IParameterValue"/>.</returns>
        IParameterValue CreateFromString(string valueText);

        /// <summary>
        /// Used mainly in grid sweepers, return the i-th distinct value for this parameter.
        /// </summary>
        IParameterValue this[int i] { get; }

        /// <summary>
        /// Used mainly in grid sweepers, return the count of distinct values for this parameter.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Returns the name of the generated parameter.
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// ID of this ValueGenerator.
        /// </summary>
        string ID { get; }
    }

    public interface INumericValueGenerator : IValueGenerator
    {
        float NormalizeValue(IParameterValue value);

        bool InRange(IParameterValue value);
    }

    public interface IDiscreteValueGenerator : IValueGenerator
    {
        double[] OneHotEncodeValue(IParameterValue value);
    }
}
