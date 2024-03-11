/*


 * added by vica, 20191120
 * */


DataLoaderClass = function () {
  const that = this;

  that.visual_encoding = {
    scatter_color:
      /*[
        '#377eb8',
        '#4daf4a',
        '#984ea3',
        '#ff7f00',
        '#ffff33',
        '#a65628',
        '#f781bf',
        '#e41a1c',
      ],
      */
      [
        '#77D1F3', // "rgb(119,209,243)",
        '#FCC32C', // "rgb(252,195,44)",
        '#89BB4C', // "rgb(137,187,76)",
        '#AB7EB5', // 'rgb(171,126,181)',
        '#6D93C4', // "rgb(109,147,196)",
        '#F78C2A', // "rgb(247, 140, 42)",
        '#E574C3', // "rgb(229, 116, 195)"
        "rgb(129,74,25)",
        "rgb(173,35,35)",
        "rgb(42,75,215)",
        "#3182bd",
        "#e6550d",
        "#31a354",
        "#756bb1",
        "#6baed6",
        "#fd8d3c",
        "#74c476",
        "#9e9ac8",
        "#9ecae1",
        "#fdae6b",
        "#a1d99b",
        "#bcbddc",
      ], //.concat(d3.schemeCategory20c),
    // http://alumni.media.mit.edu/~wad/color/palette.html
    //["rgb(255,146,51)","rgb(29,105,20)","rgb(157,175,255)",
    //"rgb(173,35,35)","rgb(129,38,192)","rgb(233,222,187)","rgb(129,197,122)","rgb(42,75,215)",
    //"rgb(41,208,208)","rgb(255,238,51)","rgb(255,205,243)","rgb(129,74,25)",
    //"#3182bd",
    //"#e6550d",
    //"#31a354",
    //"#756bb1",
    //"#6baed6",
    //"#fd8d3c",
    //"#74c476",
    //"#9e9ac8",
    //"#9ecae1",
    //"#fdae6b",
    //"#a1d99b",
    //"#bcbddc",],
    line_glyph: [
      'tri', 'circle', 'rect', 'criss', 'cross',
    ],
  };
  for (let i in that.visual_encoding.scatter_color) {
    if (that.visual_encoding.scatter_color[i][0] === 'r') {
      let x = that.visual_encoding.scatter_color[i];
      that.visual_encoding.scatter_color[i] = '#' + x.substring(4, x.length - 1).split(',').map(x => (+x).toString(16)).join('');
    }
  }
  // that.color = d3.scaleOrdinal(d3.schemeCategory10);
  // that.color = d3.scaleOrdinal(that.visual_encoding.scatter_color);
  that.color = (d) => (that.visual_encoding.scatter_color[d]);
  that.scatter_brush_area = undefined;
  that.line_brush_area = undefined;
  that.scatter_brush = undefined;
  that.line_brush = undefined;
  that.normalSize = 3;
  that.scatter_region = {};
  that.densities = [];
  that.animate_axis_time = 500;

  // state
  that.show_hover_distribution_color = false;
  that.show_gmmlabel_color = true; // show label or gmm_label

  // URL information
  that.set_dataset_url = '/setDataset';
  that.fetct_line_plot_data_url = '/linePlotData';
  that.fetch_scatter_plot_data_url = '/scatterPlotData';
  that.fetch_label_url = '/getLabel';
  that.next_data_url = '/nextData';
  that.change_win_size_url = '/changeWinSize';
  that.get_origin_url = '/getOrigin';
  that.get_grid_origin_url = '/getGridOrigin';
  that.set_tsne_attr_url = '/setTsneAttr';
  that.precompute_url = '/precompute';
  that.cache_url = '/cache';

  // data
  that.show_size = 0;
  that.current_max = 0;
  that.historical_max = 0;
  that.origin_attributes = [];
  that.total_count = 0;
  that.current_len = 0;
  that.data = [];
  that.timestamp;
  that.drift_mode = 1; // 0 for origin, 1 for compared with model, 2 for real drift
  that.all_feature_keys = [];
  that.candidate_feature_keys = []; // show in selection box
  that.selected_feature_keys = []; // show in line chart
  that.method_name = '';
  //that.overview_drift_keys = ['overview'];
  //that.selected_drift_keys = [];
  that.curData = [];
  that.curDataDic = [];
  that.actionFlags = {};
  that.scatter_scale = {
    x: undefined,
    y: undefined,
  };

  /* for density */
  that.density_config = {
    dis2den: function (c, dx, dy) {
      return c*(1 - 0.3 * (dx + dy))
      //return Math.min(c, 1)*Math.max(0, (1 - 0.3 * (dx + dy)))
    },
    den2color: function (d, r1, r2) {
      const abs = d.value > 0 ? d.value : -d.value
      return (abs / r1) ** 1.25;
    },
    range: 2,
    grid_max: Infinity,
  }

  /*
    {
        t: <time index>
        label: <gmm label>
        <attr>: <attr value>,
        x: <origin scatter plot x>,
        y: <origin scatter plot y>,
        cx: <draw scatter plot x>,
        cy: <draw scatter plot y>,
        light_color:,
        color:,
     */

  that.set_dataset = function (dataset) {
    $('#next').attr('disabled', false);
    $('#play_or_pause').attr('disabled', true);
    $("#loading")[0].style.display = "block";
    const node = new request_node(that.set_dataset_url, (data) => {
      console.log('set_dataset', data); 
      const weatherFlag = ($('#filter-select-dataset').val() == 'weather')
      if (weatherFlag) {
        that.density_config.dis2den = function (c, dx, dy) {
            //return c*(1 - 0.3 * (dx + dy))
            return Math.min(c, 1)*Math.max(0, (1 - 0.3 * (dx + dy)))
          }
        that.density_config.range = 5
      }
      that.show_size = data['current_len'];
      that.clear_views();
      that.method_name = data['method_name'];
      that.timestamp = JSON.parse(data['timestamp']);
      that.total_count = data['total_count'];
      that.current_len = data['current_len'];
      that.density_min = 0;
      that.density_max = 0;
      that.data = Array(that.current_len).fill().map(Object);
      for (let i = 0; i < that.current_len; ++i) {
        that.data[i].t = i;
        that.data[i].size = that.normalSize;
        that.data[i].timestamp = that.timestamp[i];
        that.data[i].hided = false;
      }
      that.cache_keys = [];
      that.origin_attributes = data['attributes'];
      that.all_feature_keys = data['drift_keys'];
      that.selected_feature_keys = []; // Clear previous selected keys.
      that._selected_attr = [];
      that.toggle_candidate_key(1);
      that.fetch_label();
      // Set first_time, then the animation when changing datase
      //    is same as the initialization.
      EffectForScatterplot.first_time = true;
      that.update_scatter_plot();
      // that.update_line_plot();
      // $('#filter-select-method').selectpicker('val', data['default_method']).trigger('change');
      ModelLoader.gmm_label = data['gmm_label'];
      ModelLoader.set_distributions(data['distributions']);
      ModelLoader.set_model(data['models']);
      ModelLoader.set_chunks(data['chunks']);
      that.precompute();
      ActionTrail.init();
      that.update_line_plot(function () {
        setTimeout(function () {
          //d3.select('#feature-grid-view').style('border', '1px solid lightgray');
          // d3.select('#action-trail-plot').style('border', '1px solid lightgray');
          //d3.select('#feature-grid-view').style('height', `${d3.select('.my-flex-column').node().getBoundingClientRect().height}px`);
        }, 1500);
        ActionTrail.saveHistory();
      });
      that.previous_win_size = data['win_size'];
      that.win_size = data['win_size'];
      //let slider_width = $('#win-size-slider').width()
      //let slider_height = $('#win-size-slider').height()
      //let slider = d3.sliderHorizontal()
      //  .min(data['win_size_min'])
      //  .max(data['win_size_max'])
      //  .default(data['win_size'])
      //  .width(slider_width - 30)
      //  .displayValue(true)
      //  .on('onchange', val => {
      //    that.win_size = Math.round(val);
      //  });
      //let svg = d3.select('#win-size-slider').select("svg")
      //svg.attr("width", slider_width).attr("height", slider_height)
      //svg.selectAll('*').remove()
      //svg.append('g')
      //  .attr('transform', 'translate(20,10)')
      //  .call(slider);
      //d3.selectAll('#win-size-slider .tick text').attr("y", 10);
      //d3.selectAll('#win-size-slider .parameter-value text').attr("y", 17);
      $("#loading")[0].style.display = "none";
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'dataset': dataset,
    });
    node.notify();
  };

  that.update_line_plot = function (callback) {
    if (callback) {
      that.fetch_line_plot_data(function () {
        that.draw_line_plot();
        callback();
      });
    } else
      that.fetch_line_plot_data(that.draw_line_plot);
  };

  that.update_scatter_plot = function () {
    that.show_scatter_plot();
    that.fetch_scatter_plot_data(that.draw_scatter_plot);
  };

  that.fetch_all_data = function (callback) {
    new Promise(function (resolve, reject) {
      that.fetch_label(resolve);
    }).then(function () {
      return new Promise(function (resolve, reject) {
        that.fetch_scatter_plot_data(resolve);
      });
    }).then(function () {
      return new Promise(function (resolve, reject) {
        that.fetch_line_plot_data(resolve);
      });
    }).then(function () {
      return new Promise(function (resolve, reject) {
        ModelLoader.get_distribution(resolve);
      });
    }).then(function () {
      if (that.actionFlags['new-data'] == true) {
        that.actionFlags['new-data'] = false;
        ActionTrail.notify('new-data');
      }
      if (callback !== undefined) {
        callback();
      }
    });
  };

  //that.fetch_all_data = function (callback) {
  //  that.fetch_label(function () {
  //    that.fetch_scatter_plot_data(function () {
  //      that.fetch_line_plot_data(function () {
  //        ModelLoader.get_distribution(function () {
  //          if (that.actionFlags['new-data'] == true) {
  //            that.actionFlags['new-data'] = false;
  //            ActionTrail.notify('new-data');
  //          }

  //          if (callback !== undefined) {
  //            callback();
  //          }
  //        });
  //      });
  //    });
  //  });
  //};

  that.fetch_line_plot_data = function (callback) {
    const node = new request_node(that.fetct_line_plot_data_url, (data) => {
      console.log('fetch_line_plot_data');
      if (data) {
        for (const key of data.keys) {
          for (let i = 0; i < that.current_len; ++i) {
            that.data[i][key] = data[key][i];
          }
        }
      } else {
        console.log('No data!');
      }

      // draw
      if (callback) callback();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    // node.set_data({
    //    "attributes": $('#filter-select-attribute').val().concat(
    //        $('#filter-select-driftdegree').val()),
    // });
    node.notify();
  };

  that.fetch_scatter_plot_data = function (callback) {
    const node = new request_node(that.fetch_scatter_plot_data_url, (data) => {
      console.log('fetch_scatter_plot_data');
      if (!data) {
        console.log('No data!');
        return;
      }
      //for (let i = 0; i < that.current_len; ++i) {
      //  that.data[i].x = data.x[i];
      //  that.data[i].y = data.y[i];
      //}
      console.log('data.layout', data)
      for (let i = 0; i < data.x.length; ++i) {
        that.data[i].x = data.x[i];
        that.data[i].y = data.y[i];
        if (data.layout.length > 0) {
          that.data[i].trace = data.layout[i]
//          console.log(that.data[i].trace)
        } else {
          that.data[i].trace = null
        }
      }
      // console.log(that.data)
      // that.scatter_region.xmin = data.xmin;
      // that.scatter_region.xmax = data.xmax;
      // that.scatter_region.ymin = data.ymin;
      // that.scatter_region.ymax = data.ymax;
      //const current_x = that.data.filter((d) => d.t < that.current_len).map((d) => d.x);
      //const current_y = that.data.filter((d) => d.t < that.current_len).map((d) => d.y);
      const current_x = data.x.map(d => d),
        current_y = data.y.map(d => d);
      that.scatter_region.xmin = Math.floor(Math.min(...current_x));
      that.scatter_region.xmax = Math.ceil(Math.max(...current_x));
      let x_padding = (that.scatter_region.xmax - that.scatter_region.xmin) / 20;
      that.scatter_region.xmin -= x_padding;
      that.scatter_region.xmax += x_padding;
      that.scatter_region.ymin = Math.floor(Math.min(...current_y));
      that.scatter_region.ymax = Math.ceil(Math.max(...current_y));
      let y_padding = (that.scatter_region.ymax - that.scatter_region.ymin) / 10;
      that.scatter_region.ymin -= y_padding;
      that.scatter_region.ymax += y_padding;
      if (callback) callback();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  };

  that.fetch_label = function (callback) {
    const node = new request_node(that.fetch_label_url, (data) => {
      console.log('fetch_label');
      if (!data) {
        console.log('No data!');
        return;
      }
      if (!that.show_gmmlabel_color) {
        for (let i = 0; i < data.label.length; ++i) {
          that.data[i].label = data.unique_label.indexOf(data.label[i]);
        }
      } else {
        for (let i = 0; i < data.label.length; ++i) {
          that.data[i].label = data.label[i];
        }
      }
      //for (let i = 0; i < that.current_len; ++i) {
      for (let i = 0; i < data.label.length; ++i) {
        const colors = LightenDarkenColor(that.color(that.data[i].label));
        that.data[i].color = colors[0];
        that.data[i].light_color = colors[1];
      }
      if (callback) callback();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'use_gmm_label': that.show_gmmlabel_color,
    });
    node.notify();
  };

  that.draw_scatter_plot = function () {
    that.bigSize = EffectForScatterplot.CircleVisualConfig.big_size;
    //that.normalSize = EffectForScatterplot.CircleVisualConfig.normal_size;

    // console.log("draw scatter plot");
    const div = d3.select('#scatterplot');
    const margin = {
      top: 30,
      right: 20,
      bottom: 20,
      left: 30,
    };
    let width = div.node().getBoundingClientRect().width - margin.left - margin.right;
    let height = div.node().getBoundingClientRect().height - margin.top - margin.bottom;
    // div.html(null);
    let svg = div.select('svg');
    if (svg.empty()) {
      svg = div.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);
      svg.append('g')
        .attr('class', 'scatterplot-legend')
        .attr('transform', 'translate(' + 0 + ',' + margin.top + ')');
      svg.append('g')
        .attr('class', 'density-plot')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
      svg
        .append('g')
        .attr('class', 'radios')
        .attr('transform', 'translate(' + (margin.left + width - 70) + ',' + margin.top + ')')
      svg
        .append('g')
        .attr('class', 'compared-plot')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
      svg = svg
        .append('g')
        .attr('class', 'plot')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
    } else {
      svg = div.select('svg');
      width = svg.attr('width');
      height = svg.attr('height');
      svg = svg.select('g.plot');
    }
    that.updateScatterCoords();
    that.scatter_plot_animate_time = window.EffectForScatterplot.drawNodes({
        container: svg,
        // color: that.color,
        // normalSize: that.normalSize,
        // bigSize: that.bigSize,
        // animateAxisTime: that.animate_axis_time
      },
      // that.data
      that.curData,
    );
    BrushController.update_scatter_brush();
    that.refresh_scatter_color();
    that.show_scatter_plot();
  };

  that.show_scatter_plot = function () {
    that.current_view = 'scatterplot'
    if ($(".density-plot")[0]) $(".density-plot")[0].style.display = "block"
    if ($(".plot")[0]) $(".plot")[0].style.display = "block"
    if ($(".compared-plot")[0]) $(".compared-plot")[0].style.display = "none"
    if ($(".density-hazes")[0]) $(".density-hazes")[0].style.display = "block"
    if ($(".density-heatmap")[0]) $(".density-heatmap")[0].style.display = "none"
    that.draw_view_radio()
  };

  that.show_density_diff = function () {
    that.current_view = 'density diff'
    if ($(".density-plot")[0]) $(".density-plot")[0].style.display = "none"
    if ($(".plot")[0]) $(".plot")[0].style.display = "none"
    if ($(".compared-plot")[0]) $(".compared-plot")[0].style.display = "none"
    if ($(".density-hazes")[0]) $(".density-hazes")[0].style.display = "none"
    if ($(".density-heatmap")[0]) $(".density-heatmap")[0].style.display = "block"
    that.draw_view_radio()
  }

  that.show_scatter_diff = function () {
    that.current_view = 'scatter diff'
    if ($(".density-plot")[0]) $(".density-plot")[0].style.display = "none"
    if ($(".plot")[0]) $(".plot")[0].style.display = "block"
    if ($(".compared-plot")[0]) $(".compared-plot")[0].style.display = "block"
    if ($(".density-hazes")[0]) $(".density-hazes")[0].style.display = "none"
    if ($(".density-heatmap")[0]) $(".density-heatmap")[0].style.display = "none"
    that.draw_view_radio()
  }

  that.clear_views = function () {
    d3.select('#lineplot').html(null);
    d3.select('#scatterplot').html(null);
  };

  that.draw_view_radio = function () {
    const radios = d3.select('#scatterplot').select('svg').select('g.radios')
    radios.selectAll('*').remove()
    //const views = ['scatter plot', 'density diff', 'scatter diff']
    const views = ['scatterplot', 'density diff']
    const enter = radios.selectAll('.radio')
      .data(views)
      .enter()
      .append('g')
      .attr('class', 'radio')
      .on('click', function (d) {
        if (d == 'scatterplot') {
          that.show_scatter_plot()
          ModelLoader.render_border()
          return
        }
        if (!DataLoader.density_max) {
          return;
        }
        if (d == 'scatter diff') {
          that.show_scatter_diff()
          ModelLoader.render_border()
        } else {
          that.show_density_diff()
          ModelLoader.compared_batch_idx = null
          ModelLoader.render_border()
          that.draw_density_grid()
        }
      })

    enter.append('circle')
      .attr('cx', -10)
      .attr('cy', (d, i) => i * 20-20)
      .attr('r', 8)
      .attr('stroke', '#555')
      .attr('stroke-width', 2)
      .style('fill', '#fff')

    enter.append('circle')
      .attr('cx', -10)
      .attr('cy', (d, i) => i * 20-20)
      .attr('r', 5)
      .attr('stroke', 'none')
      .attr('stroke-width', 0)
      .style('fill', (d, i) => {
        if (d == that.current_view) {
          return '#333'
        } else if (i >= 0 && !DataLoader.density_max) {
          return '#ccc'
        } else {
          return '#fff'
        }
      })

    enter.append('text')
      .attr('x', -0)
      //.attr('x', -20)
      .attr('y', (d,i) => i *20-20)
      //.attr('y', (d, i) => i * 20-20)
      .text((d) => d)
      .style('font-size', '16px')
      //.style('font-size', '22px')
      .attr('alignment-baseline', 'middle');
  };

  that.draw_line_plot = function () {
    const div = d3.select('#lineplot');
    const margin = {
      top: 10,
      //right: 60,
      right: 10,
      bottom: 15,
      left: 45,
    };

    const width = div.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = div.node().getBoundingClientRect().height - margin.top - margin.bottom;

    // initialization for linechart-svg
    let svg = div.select('svg');
    if (svg.empty()) {
      svg = div.append('svg')
    }
    svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);
    // initialization for feature-grid-svg
    let gridSVG = that.ensureFeatureGridView();

    const paddingX = 45;
    const paddingY = 10;

    // that._updateSelectedDrift();
    that.updateGridLinechart();

    const linechartDetail = window.EffectForLinechart.drawLinechart({
        container: svg,
        area: {
          x: margin.left,
          y: margin.top,
          //width: width / 2 - paddingX,
          width: width - paddingX,
          height: height - paddingY,
          //height: height / 2 - paddingY,
        },
        x_box: {
          height: 0,
          slice_num: Math.min(that.curData.length - 1, 100)
        },
        animateAxisTime: that.animate_axis_time,
        x_label: 'Time',
        y_label: '', //'Drift on All Features',
        glyph_assigner: SVGUtil.getFormatAssigner(that.visual_encoding.line_glyph),
        has_glyph: true
      },
      that.curData,
      'detail-linechart',
      that.selected_feature_keys,
      that.scatter_plot_animate_time);
    BrushController.set_line_brush([linechartDetail]);
  };

  that.updateGridLinechart = function () {
    let gridSVG = that.ensureFeatureGridView();
    window.EffectForLinechart.drawGridLinechart({
        container: gridSVG,
        area: {
          x: 0,
          y: 0,
          width: parseFloat(gridSVG.attr('width')),
          height: parseFloat(gridSVG.attr('height'))
        },
        y_label: '',
      },
      that.curData,
      'feature-overview',
      that.candiate_feature_keys,
    );
  };

  that.scatter_legend_onmouseover = (d) => {
    BrushController.refresh_scatter(ModelLoader.distributions['components'][d]['idx']);
  };
  that.scatter_legend_onmouseout = () => {
    BrushController.reset_brush();
  };
  that.scatter_legend_onclick = function (d) {
    // BrushController.reset_brush(ModelLoader.distributions['components'][d]['idx'], true);
    let hided = true;
    if (ModelLoader.distributions['components'][d]['idx'].length > 0) {
      hided = !that.data[ModelLoader.distributions['components'][d]['idx'][0]].hided;
    }
    for (let id of ModelLoader.distributions['components'][d]['idx']) {
      that.data[id].hided = hided;
    }
    //d3.select(this).style('fill', hided ? '#ffffff' : that.color(d));
    d3.select(`.legend-circle[component='${d}']`).style('fill', hided ? '#ffffff' : that.color(d));
    d3.select('#scatterplot').selectAll('.scatter').style('display', d => d.hided ? "none" : null);
    d3.selectAll(`.density[component='${d}']`).style('display', hided ? "none" : null);
    d3.selectAll(`.density-haze[component='${d}']`).style('display', hided ? "none" : null);
    if (hided) {
      d3.selectAll(`.density-add[component-1='${d}']`).style('display', hided ? "none" : null);
      d3.selectAll(`.density-add[component-2='${d}']`).style('display', hided ? "none" : null);
    } else {
      const set = Array.from(new Set(that.curData.map((x) => x.label))).sort((a, b) => a - b);
      d3.selectAll(`.density-add`).style('display', null);
      for(let i of set) {
        if (that.data[ModelLoader.distributions['components'][i]['idx'][0]].hided) {
          d3.selectAll(`.density-add[component-1='${i}']`).style('display', "none");
          d3.selectAll(`.density-add[component-2='${i}']`).style('display', "none");
        }
      }
    }
  };

  that.unhide_all_scatter = function () {
    that.data.forEach(d => d.hided = false);
    d3.select("#scatterplot").selectAll('.scatter').style('display', null);
  }

  that.refresh_scatter_color = function () {
    that._getScatterItems()
      .style('fill', function (d) {
        return BrushController.selected_idx.has(d.t) ? d.color : d.light_color;
      })
      .attr('r', (d) => BrushController.selected_idx.has(d.t) ? that.bigSize : d.size);
    const legend = d3.select(".scatterplot-legend");
    legend.html(undefined);
    //const set = Array.from(new Set(that.data.map((x) => x.label))).sort((a, b) => a - b);
    const set = Array.from(new Set(that.curData.map((x) => x.label))).sort((a, b) => a - b);
    const enter = legend.selectAll('.legend')
      .data(set)
      .enter();
    enter.append('circle')
      .attr('cx', 15)
      .attr('cy', (d, i) => i * 20 - 20)
      .attr('r', 8)
      .attr('stroke', (d) => that.color(d))
      .attr('stroke-width', 1)
      .attr('class', 'legend-circle')
      .attr('component', d => d)
      .style('fill', (d) => that.color(d))
      .on('mouseover', that.scatter_legend_onmouseover)
      .on('mouseout', that.scatter_legend_onmouseout)
      .on('click', that.scatter_legend_onclick);
    enter.append('text')
      .attr('x', 30)
      .attr('y', (d, i) => i * 20 - 20)
      .text((d) => 'COMP #' + d)
      //.text((d) => '#' + d)
      .style('font-size', '16px')
      //.style('font-size', '22px')
      .attr('alignment-baseline', 'middle');

    // for (let val of set) {
    //     legend.append("circle")
    //         .attr("cx", 20)
    //         .attr("cy", legend_y)
    //         .attr("r", 5)
    //         .style("fill", that.color(val));
    //     legend.append("text")
    //         .attr("x", 30)
    //         .attr("y", legend_y)
    //         .text(val).style("font-size", "12px")
    //         .attr("alignment-baseline", "middle");
    //     legend_y += 15;
    // }
  };

  that.toggle_label_color = function () {
    that.show_gmmlabel_color = !that.show_gmmlabel_color;
    that.fetch_label(that.refresh_scatter_color);
  };

  that._getScatterItems = function () {
    return (d3.select('#scatterplot').selectAll('.scatter'));
  };

  that.draw_all_contour = function (draw) {
    if (draw) {
      that.draw_contour(ModelLoader.distributions['keys'].map((label) => ModelLoader.distributions['components'][label]['idx']));
    } else {
      d3.selectAll('.contour').remove();
    }
  };

  that.next_data = function (callback) {
    if (that.total_count == that.current_len) {
      console.log('No more data!');
      return;
    }
    that.unhide_all_scatter();
    that.actionFlags['new-data'] = true;
    $("#loading")[0].style.display = "block";
    const node = new request_node(that.next_data_url, (data) => {
      console.log('next_data', data);
      that.show_size = ModelLoader.chunks[ModelLoader.chunks.length - 1].count + data.current_len - that.current_len;
      BrushController.selected_idx = new Set()
      BrushController.showed_idx = new Set()
      BrushController.refresh_scatter([]);
      ModelLoader.gmm_label = data.gmm_label;
      // that.data.forEach(d => d.size = that.normalSize * Math.exp((d.t - that.current_len) / that.show_size * 0.5))
      for (let i = that.current_len; i < data.current_len; ++i) {
        that.data.push({
          't': i,
          'size': that.normalSize,
          'timestamp': that.timestamp[i],
          'hided': false,
        });
      }
      that.current_len = data.current_len;
      ModelLoader.update_table(() => {
        // that.fetch_label(that.update_scatter_plot);
        // that.update_line_plot();
        that.fetch_all_data(() => {
          $("#loading")[0].style.display = "none";
          if (callback)callback();
        });
      });
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  };

  that.get_grid_origin = function (idxes, callback) {
    const node = new request_node(that.get_grid_origin_url, (data) => {
      //console.log(data);
      if (callback !== undefined) {
        callback(data);
      }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idxes': idxes,
    });
    node.notify();
  };

  that.get_origin = function (idx, callback) {
    const node = new request_node(that.get_origin_url, (data) => {
      //console.log(data);
      if (callback !== undefined) {
        callback(data);
      }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idx': idx,
      'idxes': Array.from(BrushController.selected_idx),
    });
    node.notify();
  };

  that.change_win_size = function () {
    that.previous_win_size = that.win_size;
    const node = new request_node(that.change_win_size_url, () => {
      console.log('change_wins_size');
      // update snapshot for line chart
      // update line chart
      that.update_line_plot(() => {
        ModelLoader.update_table();
      });
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'win_size': that.win_size,
    });
    node.notify();
  };

  that.set_tsne_attr = function (attrs) {
    const node = new request_node(that.set_tsne_attr_url, () => {
      console.log('set_tsne_attr');
      that.update_scatter_plot();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'attrs': attrs,
    });
    node.notify();
  };

  that.precompute = function () {
    const pre_compute_node = new request_node(that.precompute_url, () => {
      console.log('precompute finiish');
    });
    pre_compute_node.notify();
  };

  that.cache = function () {
    const pre_compute_node = new request_node(that.cache_url, () => {
      console.log('cache finiish');
    });
    pre_compute_node.notify();
  };

  that.ensureFeatureGridView = function () {
    let div = d3.select('#feature-grid-view'),
      margin = {
        left: 5,
        right: 5,
        top: 5,
        bottom: 5
      };

    const width = div.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = div.node().getBoundingClientRect().height - margin.top - margin.bottom;

    // initialization for linechart-svg
    let svg = div.select('svg')
    if (svg.empty()) {
      svg = div.append('svg')
      //.style("border", "1px solid lightgrey");
    }
    //console.log('feature-grid-view', width, height)
    svg
      .attr('width', width) // + margin.left + margin.right)
      .attr('height', height) // + margin.top + margin.bottom)
      .attr('id', 'feature-grid-svg');
    return svg;
  };

  that.updateScatterCoords = function () {
    // update coordinates in scatterplot.
    const svg = d3.select('#scatterplot').select('svg');
    const density_g = svg.select('.density-plot');
    const width = parseInt(svg.attr('width'))-20;
    const height = parseInt(svg.attr('height'))-20;
    const x = d3.scaleLinear()
      .domain([that.scatter_region.xmin, that.scatter_region.xmax])
      .range([0, width]);
    const y = d3.scaleLinear()
      .domain([that.scatter_region.ymin, that.scatter_region.ymax])
      .range([height, 0]);

    function getDist(a, b) {
      return Math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))
    }

    function getPolygonArea(x) {
      let ret = 0;
      for (let i = 1; i + 1 < x.length; ++i) {
        const a = getDist(x[i], x[0]);
        const b = getDist(x[i + 1], x[0])
        const c = getDist(x[i + 1], x[i]);
        const p = (a + b + c) / 2;
        if (p >= a && p >= b && p >= c) {
          const d = Math.sqrt(p * (p - a) * (p - b) * (p - c));
          ret += d;
        }
      }
      return ret;
    }

    function getPolygonCenter(node_list) {
      let res = [0, 0];
      node_list.forEach(node => {
        res[0] += node[0];
        res[1] += node[1];
      });
      res[0] /= node_list.length;
      res[1] /= node_list.length;
      return res;
    };

    that.scatter_scale = {
      x,
      y,
    };
    //console.log(that.curData)
    that.curData.forEach((item) => {
      item.cx = x(item.x);
      item.cy = y(item.y);
      // item.trace = []
      item.converted_trace = null
      if (item.trace && item.trace.length > 0) {
        item.converted_trace = item.trace.map((d) => ({ 
          x: Math.min(Math.max(0, x(d.x)), width),
          y: Math.min(Math.max(0, y(d.y)), height)
        }) )
        // console.log('0', item.trace)
        // console.log('1', item.converted_trace)
      }
    });
    that.draw_contour = function (idx_list) {
      return;
      let densities = [];
      for (const idx of idx_list) {
        const density = d3.contourDensity()
          .x((d) => x(d.x))
          .y((d) => y(d.y))
          .size([width, height])
          .thresholds([0.05, 0.1, 0.2, 0.3])
          // .bandwidth(20)(idx.map((id => that.data[id])));
          .bandwidth(20)(idx.map((id) => that.curData[id]).filter((d) => d));

        densities = [...densities, ...density];
      }
      svg
        .selectAll('.contour')
        .data(densities)
        .enter()
        .append('path')
        .attr('class', 'contour')
        .attr('d', d3.geoPath())
        .attr('fill', 'none')
        .attr('stroke', '#87CEFA')
        .attr('stroke-linejoin', 'round');
    };

    that.draw_density = function (key, idx, color) {
      if (idx.length < 30) return;
      const density = d3.contourDensity()
        .x((d) => x(d.x))
        .y((d) => y(d.y))
        .size([width, height])
        //.thresholds([0.015, 0.1, 0.2, 0.3])
        //.thresholds([0.1, 0.2, 0.3])
        .thresholds((DataLoader.data.length < 500) ? [0.015, 0.1, 0.2] : (DataLoader.data.length < 5000) ? [0.1, 0.2, 0.3] : [0.15, 0.25, 0.35])
        .bandwidth(20)(idx.map((id) => that.data[id]));

      for (let i = 0; i < density.length; ++i) {
        density[i].color = color;
        density[i].opacity = i == 0 ? 0.2 : 0.1;
        density[i].is_first = i == 0;
        density[i].key = key;
      }
      that.density_labels.push([key, color]);
      that.densities = [...that.densities, ...density.slice(0, 1)]
    };

    that.enable_grid_normalize = function () {
      that.use_grid_normalize = true
      that.draw_density_grid()
    }

    that.disable_grid_normalize = function () {
      that.use_grid_normalize = false
      that.draw_density_grid()
    }

    that.draw_density_grid = function () {
      let old_points
      if (!ModelLoader.compared_batch_idx) {
        old_points = that.data_points.filter(d => d.t < DataLoader.density_max)
      } else {
        old_points = that.data_points.filter(d => ModelLoader.compared_batch_idx.includes(d.t))
      }
      const new_points = that.data_points.filter(d => d.t >= DataLoader.density_max)

      const grid = 10
      const grid_margin = 2
      const width = Math.max(...that.data_points.map(d => d.cx))
      const height = Math.max(...that.data_points.map(d => d.cy))
      const wn = Math.floor(width / grid) + 1
      const hm = Math.floor(height / grid) + 1
      //const count0 = [], count1 = [], idxs0 = [], idxs1 = [], diff = []
      //for (let i = 0; i < wn; ++i) {
      //  for (let j = 0; j < hm; ++j) {
      //    count0[i * hm + j] = 0
      //    count1[i * hm + j] = 0
      //    idxs0[i * hm + j] = []
      //    idxs1[i * hm + j] = []
      //  }
      //}

      //if (that.use_grid_normalize) {
      //  for (let i = 0; i < old_points.length; ++i) {
      //    const x = Math.floor(old_points[i].cx / grid)
      //    const y = Math.floor(old_points[i].cy / grid)
      //    count0[x * hm + y] += 1.0 / old_points.length
      //    idxs0[x * hm + y].push(old_points[i].t)
      //  }
      //  for (let i = 0; i < new_points.length; ++i) {
      //    const x = Math.floor(new_points[i].cx / grid)
      //    const y = Math.floor(new_points[i].cy / grid)
      //    count1[x * hm + y] += 1.0 / new_points.length
      //    idxs1[x * hm + y].push(new_points[i].t)
      //  }
      //} else {
      //  for (let i = 0; i < old_points.length; ++i) {
      //    const x = Math.floor(old_points[i].cx / grid)
      //    const y = Math.floor(old_points[i].cy / grid)
      //    count0[x * hm + y] += 1.0
      //    idxs0[x * hm + y].push(old_points[i].t)
      //  }
      //  for (let i = 0; i < new_points.length; ++i) {
      //    const x = Math.floor(new_points[i].cx / grid)
      //    const y = Math.floor(new_points[i].cy / grid)
      //    count1[x * hm + y] += 1.0
      //    idxs1[x * hm + y].push(new_points[i].t)
      //  }
      //}

      function _collec_points(points) {
        const tempCount = [],
          tempIndex = [],
          norm = (that.use_grid_normalize) ? (1 / points.length) : 1;
        for (let i = 0; i < wn; ++i) {
          for (let j = 0; j < hm; ++j) {
            tempCount[i * hm + j] = 0
            tempIndex[i * hm + j] = []
          }
        }
        for (let i = 0; i < points.length; ++i) {
          const x = Math.floor(points[i].cx / grid)
          const y = Math.floor(points[i].cy / grid)
          tempCount[x * hm + y] += norm;
          tempIndex[x * hm + y].push(points[i].t)
        }
        return [tempCount, tempIndex]
      }

      function _calc_density(count) {
        const tempDen = [];
        for (let i = 0; i < wn; ++i) {
          for (let j = 0; j < hm; ++j) {
            tempDen[i * hm + j] = 0
          }
        }
        for (let i = 0; i < wn; ++i) {
          for (let j = 0; j < hm; ++j) {
            let sum = 0;
            for (let k = i - k_dist; k <= i + k_dist; ++k) {
              if (k < 0 || k >= wn) continue;
              let d = Math.abs(i - k)
              for (let l = j - k_dist + d; l <= j + k_dist - d; ++l) {
                if (l < 0 || l >= hm) continue;
                //sum += count[k * hm + l] * (1.0 - 0.3 * (d + Math.abs(j - l)))
                sum += that.density_config.dis2den(count[k * hm + l], d, Math.abs(j - l));
              }
            }
            tempDen[i * hm + j] = Math.min(sum, that.density_config.grid_max);
          }
        }
        return tempDen;
      }

      const [count0, idxs0] = _collec_points(old_points),
        [count1, idxs1] = _collec_points(new_points);

      const k_dist = that.density_config.range
      const den0 = _calc_density(count0),
        den1 = _calc_density(count1)

      const diff = []
      for (let i = 0; i < wn; ++i) {
        for (let j = 0; j < hm; ++j) {
          let sum = 0
          let neighbors = []
          sum = den1[i * hm + j] - den0[i * hm + j]
          // for (let k = i - k_dist; k <= i + k_dist; ++k) {
          //   if (k < 0 || k >= wn) continue;
          //   let d = Math.abs(i - k)
          //   for (let l = j - k_dist + d; l <= j + k_dist - d; ++l) {
          //     if (l < 0 || l >= hm) continue;
          //     sum += (count1[k * hm + l] - count0[k * hm + l]) * (1.0 - 0.3 * (d + Math.abs(j - l)))
          //   }
          // }
          neighbors = neighbors.concat(...idxs0[i * hm + j], ...idxs1[i * hm + j])
          diff[i * hm + j] = {
            value: sum,
            neighbors: neighbors
          }
        }
      }
      const max_value = Math.max(...diff.map(d => d.value > 0 ? d.value : -d.value))
      const maxDen0 = Math.max(...den0),
        maxDen1 = Math.max(...den1);
      console.log('------------ debug: ', max_value, maxDen0, maxDen1)

      const heatmap = svg.select('g.density-addition')
        .select('g.density-heatmap')

      heatmap.selectAll(".heat")
        .data(diff).enter()
        .append("rect")
        .attr("class", "heat")
        .attr("x", (d, i) => Math.floor(i / hm) * grid)
        .attr("y", (d, i) => i % hm * grid)
        .attr("width", grid - grid_margin)
        .attr("height", grid - grid_margin)
        .attr("fill", d => d.value > 0 ? 'rgb(204, 98, 87)' : '#5787cc')
        .style("stroke", "none")
        .style("fill-opacity", 0.01)
        .style("stroke-width", .2)

      heatmap.selectAll(".heat")
        .data(diff)
        .attr("fill", d => d.value > 0 ? 'rgb(204, 98, 87)' : '#5787cc')
        .style("fill-opacity", d => {
          //  const abs = d.value > 0 ? d.value : -d.value
          //  return (abs / max_value) ** 1.25
          return that.density_config.den2color(d, max_value, Math.max(maxDen0, maxDen1));
        })
        .on('mouseover', function (d) {
          that.handleGridMouseOver(d, d3.select(this));
        })
        .on('mouseout', function (d) {
          that.handleGridMouseOut(d, d3.select(this));
        });

    }

    that.draw_batch_density = function (data_points, duration, delay) {
      that.data_points = data_points
      const old_points = that.data_points.filter(d => d.t < DataLoader.density_max)

      delay = delay || 0
      that.prev_densities = that.densities
      that.densities = []
      that.density_labels = []

      const set = Array.from(new Set(old_points.map((x) => x.label))).sort();
      for (const key of set) {
        that.draw_density(key, old_points.filter((d) => d.label == key).map((d) => d.t),
          LightenDarkenColor(that.color(key))[1]);
      }
      that.density_transition(density_g, duration, delay);

      svg.select('.density-labels').remove();
      const labels = svg.append("g")
        .attr("class", "density-labels");
      let slc = labels.selectAll('.density-label')
        .data(that.density_labels).enter()
        .append("defs")
        .attr("class", "density-label")
        .append("radialGradient")
        .attr("id", (d, i) => "gradient" + d[0]);
      slc.append("stop").attr("offset", "30%")
        .attr("stop-color", d => `rgba(${parseInt(d[1].slice(1,3), 16)},${parseInt(d[1].slice(3,5), 16)},${parseInt(d[1].slice(5,7), 16)},0.1)`)
      slc.append("stop").attr("offset", "95%")
        .attr("stop-color", d => `rgba(${parseInt(d[1].slice(1,3), 16)},${parseInt(d[1].slice(3,5), 16)},${parseInt(d[1].slice(5,7), 16)},0.01)`)

      svg.select(".density-addition").remove()
      const addition = svg.append("g")
        .attr("class", "density-addition")
        .attr("transform", "translate(30,30)")

      const haze = addition.append("g")
        .attr("class", "density-hazes")

      const heatmap = addition.append("g")
        .attr("class", "density-heatmap")
        .style("display", "none")

      that.draw_density_grid()

      let counter = 0;
      let data = []
      for (const key of set) {
        let idx = old_points.filter((d) => d.label == key).map(d => ({
          x: d.x,
          y: d.y,
          k: key
        }))
        data = data.concat(idx);
        ++counter;
      }

      for (let i = 0; i < that.densities.length; ++i) {
        for (let j = i + 1; j < that.densities.length; ++j) {
          let intersec = turf.intersect(that.densities[i], that.densities[j])
          if (intersec) {
            haze.append('path')
              .attr('class', 'density-add')
              .attr('component-1', i)
              .attr('component-2', j)
              .attr('d', d3.geoPath()(intersec))
              .attr('fill', that.densities[i].color)
              .attr('fill-opacity', 0)
              .transition().duration(duration / 2).delay(delay + duration / 2)
              .attr('fill-opacity', 0.1)
            haze.append('path')
              .attr('class', 'density-add')
              .attr('component-1', i)
              .attr('component-2', j)
              .attr('d', d3.geoPath()(intersec))
              .attr('fill', that.densities[j].color)
              .attr('fill-opacity', 0)
              .transition().duration(duration / 2).delay(delay + duration / 2)
              .attr('fill-opacity', 0.1)
          }
        }
      }

      let sample_threshold = 250 / data.length
      let sample_data = data.filter(d => Math.random() <= sample_threshold)
      haze.selectAll('.density-haze')
        .data(sample_data).enter()
        .append("circle")
        .attr("class", "density-haze")
        .attr("component", d=>d.k)
        .attr("cx", d => that.scatter_scale.x(d.x))
        .attr("cy", d => that.scatter_scale.y(d.y))
        .attr("r", 18)
        .attr("fill", d => `url('#gradient${d.k}')`)
        .style("opacity", 0)
        .style("stroke", "none")

      haze.selectAll('.density-haze')
        .data(sample_data)
        .transition().duration(duration / 2).delay(delay + duration / 2)
        .style("opacity", 1)

      //const last_batch = data_points.filter(d => d.t >= DataLoader.density_min && d.t < DataLoader.density_max)
      //const R = 10
      //const H = Math.sqrt(R*1.25)
      //let compared_view = svg.select("g.compared-plot")
      //compared_view.selectAll('*').remove()
      //that.compared_plot = compared_view.selectAll(".glyph")
      //  .data(last_batch).enter()
      //  .append('g')
      //  .attr('class', 'glyph')
      //  .attr('transform', d => `translate(${d.cx},${d.cy})`)

      //that.compared_plot.append('path')
      //  .attr('d', d => `M0 ${-H}L${-R/2} ${H}L${R/2} ${H}Z`)
      //  .style('fill-opacity', .5)
      //  .style('stroke', 'gray')
      //  .style('stroke-width', 0.35)
      //  .style('fill', d => d.light_color)
      //  .on('mouseover', function(d){
      //    d3.select(this).style('stroke-width', 1).style('fill-opacity', 1)
      //  })
      //  .on('mouseout', function(d){
      //    d3.select(this).style('stroke-width', 0.35).style('fill-opacity', 0.5)
      //  })
    };

    that.density_transition = function (svg, duration, delay) {
      //console.log('density_transition --- ', that.historical_max, that.curData.length);
      const den2key = (d) => ('' + d.key + '_' + d.opacity.toFixed(3));
      const getDist2 = (a, b) => ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));

      let polygons = [],
        key2density = {};
      that.prev_densities.forEach((density) => {
        key2density[den2key(density)] = density;
      });

      that.densities.forEach((density) => {
        const oldDensity = key2density[den2key(density)];
        if (oldDensity !== undefined) {
          // match polygons between old density and current density.
          let oldCenters = oldDensity.coordinates.map(d => getPolygonCenter(d[0])),
            oldAreas = oldDensity.coordinates.map(d => getPolygonArea(d[0]));
          density.coordinates.forEach((polygon, idInDensity) => {
            let center = getPolygonCenter(polygon[0]),
              area = getPolygonArea(polygon[0]);
            let minIndex = undefined,
              minDis = undefined;
            oldCenters.forEach((oc, idx) => {
              if (oldAreas[idx] < 1e-5) {
                return;
              }
              let areaRatio = (area + 1e-5) / oldAreas[idx];
              areaRatio = Math.max(areaRatio, 1 / areaRatio);
              const dis = getDist2(oc, center) * areaRatio;
              if (minIndex === undefined || dis < minDis) {
                minDis = dis;
                minIndex = idx;
              }
            });
            polygons.push({
              density,
              index: idInDensity,
              prevDensity: oldDensity,
              matchedIndex: minIndex
            });
          });
        } else {
          density.coordinates.forEach((polygon, idInDensity) => {
            polygons.push({
              density,
              index: idInDensity
            });
          });
        }
      });

      polygons.forEach((p) => {
        p.renderData = Object.assign({}, p.density);
        p.renderData.coordinates = [p.density.coordinates[p.index]];
        if (p.prevDensity !== undefined && p.matchedIndex !== undefined) {
          p.prevRenderData = Object.assign({}, p.prevDensity);
          p.prevRenderData.coordinates = [p.prevDensity.coordinates[p.matchedIndex]];
        }
      });

      svg.selectAll('.density').classed('density', false)
        .transition().delay(delay).duration(10).remove();
      // Change by kelei: A current visual item (class=density) is a polygon of density.
      let slc = svg.selectAll('.density').data(polygons, function (d) {
        return ('' + d.density.key + '_' + d.density.opacity.toFixed(3) + '_' + d.index);
      });
      //console.log('refactored density rendering', polygons);

      let slcEnter = slc.enter().append('path')
        .classed('density', true)
        .attr('component', d => d.density.key)
        //.attr('fill-opacity', 0)
        //.attr('fill', d => ((d.prevRenderData === undefined) ? d.density.color : d.prevRenderData.color))
        //.attr('d', d => ((d.prevRenderData === undefined) ? d3.geoPath()(d.renderData) : d3.geoPath()(d.prevRenderData)))
        .style('pointer-events', 'none');

      const isFromOld = (d) => (d.prevRenderData !== undefined);
      let slcEnterNew = slcEnter.filter(d => (!isFromOld(d))),
        slcEnterTran = slcEnter.filter(isFromOld);

      const _update = (_slc) => (_slc.attr('fill', d => d.density.color)
        .attr('fill-opacity', d => d.density.opacity)
        .attr('stroke', d => d.density.is_first ? 'none' : 'lightgray')
        .attr('stroke-width', 0.3));

      slcEnterNew = slcEnterNew.style('opacity', 0)
        .attr('fill-opacity', d => d.renderData.opacity)
        .attr('fill', d => d.renderData.color)
        .attr('d', d => d3.geoPath()(d.renderData))
        .transition().delay(delay).duration(duration).style('opacity', 1);

      slcEnterTran = slcEnterTran.attr('fill-opacity', d => d.prevRenderData.opacity)
        .attr('fill', d => d.renderData.color)
        .attr('d', d => d3.geoPath()(d.prevRenderData))
        .attr('stroke', d => d.prevRenderData.is_first ? 'none' : 'lightgray')
        .attr('stroke-width', 0.3)
        .transition().delay(delay).duration(duration)
        .attrTween('d', function (d) {
          const begin = d3.geoPath()(d.prevRenderData),
            end = d3.geoPath()(d.renderData)
          return d3.interpolatePath(begin, end);
        });

      _update(slcEnterNew);
      _update(slcEnterTran);
    };

    // original render process for density
    that._density_transition = function (svg, duration) {
      svg
        .selectAll('.density')
        .data(that.densities)
        .enter()
        .append('path')
        .attr('class', 'density')
        .attr('component', d => d.key)
        .attr('d', d3.geoPath())
        .attr('fill', d => d.color)
        .attr('fill-opacity', 0)
        .style('pointer-events', 'none');

      svg
        .selectAll('.density')
        .data(that.densities)
        .transition().duration(duration)
        .attrTween('d', function (d) {
          var previous = d3.select(this).attr('d');
          // console.log(previous)
          //var current = d3.geoPath()(d);
          //return d3.interpolatePath(previous, current);
          let newD = Object.assign({}, d);
          if (newD.coordinates.length > 0) {
            let largest = undefined;
            newD.coordinates.forEach((p) => {
              if (largest === undefined || p.length > largest.length) {
                largest = p;
              }
            });
            newD.coordinates = [largest];
          }
          const internal = d3.geoPath()(newD),
            current = d3.geoPath()(d),
            timeStep1 = 0.3;
          const interpolateStep1 = d3.interpolatePath(previous, internal),
            interpolateStep2 = d3.interpolatePath(internal, current);
          return (function (t) {
            if (t < timeStep1) {
              return interpolateStep1(t / timeStep1);
            } else {
              return interpolateStep2((t - timeStep1) / (1 - timeStep1));
            }
          });
        })
        .attr('fill', d => d.color)
        .attr('fill-opacity', d => d.opacity)
        .attr('stroke', d => d.is_first ? 'none' : 'lightgray')
        .attr('stroke-width', 0.3);

      svg
        .selectAll('.densities')
        .data(that.densities).exit().remove()
        .transition().duration(300)
        .attr('fill-opacity', 0);

    }
  };

  that.getVisibleData = function (data) {
    return data; //.filter((d) => (d.t > that.historical_max));
  };

  that.toggle_candidate_key = function (drift_mode) {
    //let selected_attr = that.selected_feature_keys.filter(x => x.includes('_')).map(x => x.split('_')[0]);
    let selected_attr = that.selected_feature_keys.filter(x => x.includes('_')).map(x => x.split('_')[0]);
    let contains_overview = selected_attr.length < that.selected_feature_keys.length;
    that.drift_mode = drift_mode;
    if (that.drift_mode == 0) {
      that.candiate_feature_keys = that.origin_attributes.map(x => `${x}_Origin`);
      //that.selected_feature_keys = selected_attr.map(x => `${x}_Origin`);
      that.selected_feature_keys = that._selected_attr.map(x => `${x}_Origin`);
    }
    if (that.drift_mode == 1) {
      that._selected_attr = selected_attr.map(x => x);
      that.candiate_feature_keys = that.origin_attributes.map(x => `${x}_${that.method_name}`);
      that.candiate_feature_keys.splice(0, 0, that.method_name);
      that.selected_feature_keys = ['ED'];
      //that.selected_feature_keys = selected_attr.map(x => `${x}_${that.method_name}`);
      //that.selected_feature_keys.splice(0, 0, that.method_name);
      // if (that.method_name == 'ED') {
      //   that.selected_feature_keys.splice(0, 0, 'ED2');
      // }
    }
    //if (that.drift_mode == 2) {
    //  that.candiate_feature_keys = that.origin_attributes.map(x => `${x}_${that.method_name}2`);
    //  that.candiate_feature_keys.splice(0, 0, that.method_name + '2');
    //  that.selected_feature_keys = selected_attr.map(x => `${x}_${that.method_name}2`);
    //  that.selected_feature_keys.splice(0, 0, that.method_name + '2');
    //}
  }

  that.handleGridMouseOver = function (d, vn) {
    d._pop_show = true;
    const genTriggerPopover = function () {
      return function (data) {
        //console.log(data)
        if (d._pop_over === undefined) {
          $(vn.node()).popover({
            container: 'body',
            trigger: 'manual',
            placement: 'top',
            html: true
          });
          d._pop_over = true;
        }
        if (d._pop_show) {
          let html = "";
          for (let i = 0; i < data.idxes.length; ++i) {
            let row = `${data.idxes[i]}: `
            if (data.values[i].text != undefined) {
              row += data.values[i].text;
            } else {
              let is_first = true;
              for (let key in data.values[i]) {
                if (typeof (data.values[i][key]) == "number") {
                  row += `${is_first ? '' : ', '}` + key + ': ' + Number(data.values[i][key]).toFixed(2);
                } else {
                  row += `${is_first ? '' : ', '}` + key + ': ' + data.values[i][key];
                }
                is_first = false;
              }
            }
            html += `<p>${row}</p>`
          }
          vn.attr('data-content', html);
          $(vn.node()).popover('show');
        }
      };
    };
    DataLoader.get_grid_origin(d.neighbors, genTriggerPopover(vn));
  };

  that.handleGridMouseOut = function (d, vn) {
    d._pop_show = false;
    if (d._pop_over) {
      $(vn.node()).popover('hide');
    }
  };

};

/*

    that.density_transition = function(svg, duration) {
      const n = Math.max(that.densities.length, that.prev_densities.length);
      for (let i = 0; i < n; ++i) {
        if (i >= that.densities.length || that.densities[i].coordinates.length == 0) {
          // disappear
          if (i < that.prev_densities.length && that.prev_densities[i].g) {
            that.prev_densities[i].g
              .exit().remove()
              .transition().duration(300)
              .attr('fill-opacity', 0);
          }
          that.densities[i].g = null;
        } else if (i >= that.prev_densities.length || that.prev_densities[i].coordinates.length == 0) {
          // appear
          const d = that.densities[i];
          that.densities[i].g =
            svg.append('g')

          that.densities[i].g.selectAll('.density')
              .data(that.densities[i].coordinates).enter()
              .append('path')
              .attr('class', 'density')
              .attr('d', d => 'M' + d.map(e=>`${e[0]},${e[1]}`).join('L') + 'Z')
              .attr('fill', d.color)
              .attr('fill-opacity', 0)
              .style('pointer-events', 'none');

          that.densities[i].g.selectAll('.density')
              .data(that.densities[i].coordinates)
              .transition().duration(duration)
              .attr('fill-opacity', d.opacity)
              .attr('stroke', d.is_first ? 'none': 'lightgray')
              .attr('stroke-width', 0.3);
        } else {
          const d = that.densities[i];
          that.densities[i].g = that.prev_densities[i]
              .g.selectAll('.density')
              .data(that.densities[i].coordinates)
              .transition().duration(duration)
              .attrTween('d', function(d) {
                var previous = d3.select(this).attr('d');
                // console.log(previous)
                var current = 'M' + d.map(e=>`${e[0]},${e[1]}`).join('L') + 'Z';
                const interpolate = d3.interpolatePath(previous, current);
                console.log(interpolate);
                return interpolate;
              })
              .attr('fill', d.color)
              .attr('fill-opacity', d.opacity)
              .attr('stroke', d.is_first ? 'none': 'lightgray')
              .attr('stroke-width', 0.3);
        }
      }
      */