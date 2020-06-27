// <copyright file="AutoMLTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.Expert.Tests
{
    public class AutoMLTest : TestBase
    {
        public AutoMLTest(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void ColumnPicker_should_create_from_dataview()
        {
            var dataview = this.GetDataView<TestDataView1>("test_dataview.csv");
            var option = new ColumnPicker.Option()
            {
                LabelColumn = "Label",
                CatagoricalColumns = new string[] { "TokenColumn1", "TokenColumn2" },
            };
            var columnPicker = new ColumnPicker(dataview, option);

            columnPicker.LabelColumn.Name.Should().Be("Label");
            columnPicker.NumericColumns.Select(x => x.Name)
                        .Should().HaveCount(2)
                        .And
                        .Contain("NumericColumn1")
                        .And
                        .Contain("NumericColumn2");

            columnPicker.TextColumns.Select(x => x.Name)
                        .Should().HaveCount(2)
                        .And
                        .Contain("TextColumn1")
                        .And
                        .Contain("TextColumn2");

            columnPicker.CatagoricalColumns.Select(x => x.Name)
                        .Should().HaveCount(2)
                        .And
                        .Contain("TokenColumn1")
                        .And
                        .Contain("TokenColumn2");

            columnPicker.AvailableColumns.Should().HaveCount(6);
        }

        [Fact]
        public void ColumnPicker_should_ignore_columns_in_IgnoreColumns_option()
        {
            var dataview = this.GetDataView<TestDataView1>("test_dataview.csv");
            var option = new ColumnPicker.Option()
            {
                LabelColumn = "Label",
                CatagoricalColumns = new string[] { "TokenColumn1", "TokenColumn2" },
                IgnoreColumns = new string[] { "TokenColumn1", "NumericColumn1", "TextColumn1"},
            };
            var columnPicker = new ColumnPicker(dataview, option);

            columnPicker.LabelColumn.Name.Should().Be("Label");
            columnPicker.NumericColumns.Select(x => x.Name)
                        .Should().HaveCount(1)
                        .And
                        .Contain("NumericColumn2");

            columnPicker.TextColumns.Select(x => x.Name)
                        .Should().HaveCount(1)
                        .And
                        .Contain("TextColumn2");

            columnPicker.CatagoricalColumns.Select(x => x.Name)
                        .Should().HaveCount(1)
                        .And
                        .Contain("TokenColumn2");

            columnPicker.AvailableColumns.Should().HaveCount(3);
        }

        [Fact]
        public void ColumnPicker_should_select_column_from_available_columns()
        {
            var dataview = this.GetDataView<TestDataView1>("test_dataview.csv");
            var option = new ColumnPicker.Option()
            {
                LabelColumn = "Label",
                CatagoricalColumns = new string[] { "TokenColumn1", "TokenColumn2" },
                IgnoreColumns = new string[] { "TokenColumn1", "NumericColumn1", "TextColumn1" },
            };
            var columnPicker = new ColumnPicker(dataview, option);

            var selectedColumn = columnPicker.SelectColumn(new DataViewSchema.Column[] { }, 1);
            selectedColumn.Count().Should().Be(1);
            selectedColumn.First().Name.Should().BeOneOf(columnPicker.AvailableColumns.Select(x => x.Name).ToArray());
        }

        [Fact]
        public void ColumnPicker_should_not_select_column_from_selected_columns()
        {
            var dataview = this.GetDataView<TestDataView1>("test_dataview.csv");
            var option = new ColumnPicker.Option()
            {
                LabelColumn = "Label",
                CatagoricalColumns = new string[] { "TokenColumn1", "TokenColumn2" },
            };
            var columnPicker = new ColumnPicker(dataview, option);

            var selectedColumn = columnPicker.SelectColumn(new DataViewSchema.Column[] { }, 2);
            var selectedColumn2 = columnPicker.SelectColumn(selectedColumn, 5);
            selectedColumn2.Should().HaveCount(4);
            foreach (var column in selectedColumn2)
            {
                column.Name.Should().BeOneOf(columnPicker.AvailableColumns.Select(x => x.Name));
                selectedColumn.Select(x => x.Name).Should().NotContain(column.Name);
            }
        }

        [Fact]
        public void ColumnPicker_should_return_column_type()
        {
            var dataview = this.GetDataView<TestDataView1>("test_dataview.csv");
            var option = new ColumnPicker.Option()
            {
                LabelColumn = "Label",
                CatagoricalColumns = new string[] { "TokenColumn1", "TokenColumn2" },
            };
            var columnPicker = new ColumnPicker(dataview, option);
            columnPicker.GetColumnType("TokenColumn1").Should().Be(ColumnType.Catagorical);
            columnPicker.GetColumnType("TextColumn1").Should().Be(ColumnType.String);
            columnPicker.GetColumnType("NumericColumn1").Should().Be(ColumnType.Numeric);
        }

        private IDataView GetDataView<T>(string fileName)
        {
            var context = new MLContext();
            var file = this.GetFileFromTestData(fileName);
            var dataview = context.Data.LoadFromTextFile<T>(file, separatorChar: ',', hasHeader: true);
            return dataview;
        }

        private class TestDataView1
        {
            [LoadColumn(0)]
            public string Label;

            [LoadColumn(1)]
            public string TextColumn1;

            [LoadColumn(2)]
            public string TextColumn2;

            [LoadColumn(3)]
            public float NumericColumn1;

            [LoadColumn(4)]
            public float NumericColumn2;

            [LoadColumn(5)]
            public string TokenColumn1;

            [LoadColumn(6)]
            public string TokenColumn2;
        }
    }
}
