/*
 * added by vica, 20191123
 * */


ModelLoaderClass = function () {
  const that = this;

  // that.color = d3.scaleOrdinal(d3.schemeCategory10);

  // state
  that.show_gmmlabel_color = false; // show label or gmm_label
  //
  // URL information
  that.add_model_url = '/addModel';
  that.update_model_url = '/updateModel';
  that.update_table_url = '/updateTable';
  that.adjust_component_url = '/adjustComponent';
  that.get_distributions_url = '/getDistributions';
  that.adapt_url = '/adapt';
  that.single_chunk_info_url = '/singleChunkInfo';
  that.get_predict_vector_url = '/getPredictVector';
  that.get_predict_stat_url = '/getPredictStat';
  that.get_square_data_url = '/getSquareData';

  // data
  that.models = null;
  that.chunks = null;
  that.distributions = null;
  that.model_colors = [...Array(12).keys()].map(x => `url(#pattern${x})`);
  that.PerfStatus = null;
  that.performance_comparsion = false;

  that.set_model = function (models) {
    that.models = models;
    d3.select('#model-tbody').html(null);
    const rows = d3.select('#model-tbody')
      .selectAll('tr')
      .data(that.models)
      .enter()
      .append('tr')
      .on('mouseover', that.onmouseover)
      .on('mouseout', that.onmouseout)
      .on('click', that.onclick);
    that._createModelTableTD(rows);

    // $('.model-toggle').bootstrapToggle();
    // for (let d of that.models) {
    //     $(`#model_${d.id}`).bootstrapToggle(d.use ? 'on' : 'off');
    // }
    // $('.model-toggle').change(that.toggle_model);
  };

  that.toggle_model = function (id, checked) {
    const idx = that.models.findIndex((x) => x.id == id);
    that.models[idx].use = checked;
    const selectedInfo = {};
    that.models.forEach((d)=>{
      selectedInfo[d.id] = d.use;
    });
    // that.models[idx].use = !that.models[idx].use;

    const node = new request_node(that.update_model_url, () => {
      // const exists_model = that.models.map((x) => x.use).reduce((a, b) => a || b, false);
      // if (exists_model) {
      //   // that.update_table(DataLoader.update_line_plot);
      // }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'id': id,
      'all_use': selectedInfo
    });
    node.notify();
    //event.preventDefault();
    event.stopPropagation();
  };

  that.set_chunks = function (chunks) {
    that.chunks = chunks;
    d3.select('#drift-current').html(null);
    d3.select('#drift-tbody').html(null);
    d3.select('#drift-current')
      .append('tr')
      .attr("class", "content")
    d3.select('#drift-tbody')
      .selectAll('tr')
      .data(that.chunks.slice(1))
      .enter()
      .append('tr')
      .attr("class", "content")

    const rows = d3.select('#drift-table')
      .selectAll('tr.content')
      .data(that.chunks)
      .on('mouseover', that.onmouseover)
      .on('mouseout', that.onmouseout)
      .on('click', that.onclick)

    const cells = rows.selectAll('td')
      .data((d, i) => [i, d['count'], my_round(Math.abs(d['drift']))])
      .enter()
      .append('td')
      .style('text-align', 'center')
      .style('vertical-align', (d, i) => i == 0 && d <= 1 ? 'bottom' : 'middle')
      .text(function (d, i) {
        if (i === 0) {
          if (d == 0) {
            SVGUtil.styleD3(d3.select(this), {
              'font-size': '24px',
              'padding-top': '18px',
              'padding-bottom': '0px',
            });
            return '*';
          } else {
            return (d - 1);
          }
        }
        return d;
      });
    rows.append('td').attr('align', 'center').each(function (d) {
      that.draw_percentage(d3.select(this), d['gaussian_percentage'], Array.from(new Set(that.gmm_label)).sort((a, b) => a - b).map(x => DataLoader.color(x)));
    });
    rows.append('td').attr('align', 'center').each(function (d) {
      that.draw_percentage(d3.select(this), d['model_percentage']);
    });

    const current_offset = $("#drift-current").offset()
    const table_offset = $("#drift-table").offset()
    const tbody_offset = $("#drift-tbody").offset()

    d3.select("#current_label")
      .style("top", `${current_offset.top - table_offset.top - 10}px`)
      .style("display", "block")

    d3.select("#streaming_label")
      .style("top", `${tbody_offset.top - table_offset.top - 10}px`)
      .style("display", "block")
  };

  that.set_introduction = function (span) {
    return;
    span.style('float', 'right');
    let text = span.append('p').text('Current Selection')
    SVGUtil.styleD3(text, {
      'font-size': '14px',
      'margin-left': '12px',
      display: 'inline',
      'font-weight': 'bold'
    });
    let tag = span.append('p').text('*');
    SVGUtil.styleD3(tag, {
      'font-size': '20px',
      'vertical-align': 'top',
      display: 'inline',
      'margin-left': '5px'
    });
    let text2 = span.append('p').text('Streaming Data')
    SVGUtil.styleD3(text2, {
      'font-size': '14px',
      'margin-left': '45px',
      display: 'inline',
      'font-weight': 'bold'
    });
    let text3 = span.append('p').text('0,1,2,...');
    SVGUtil.styleD3(text3, {
      'font-size': '14px',
      'margin-left': '5px',
      display: 'inline'
    });
  };

  that.adjust_component = function (idxes) {
    const node = new request_node(that.adjust_component_url, (data) => {
      console.log('merge component');
      if (data) {
        that.gmm_label = data['gmm_label'];
        that.set_chunks(data['chunks']);
        that.set_model(data['models']);
        DataLoader.update_line_plot();
      } else {
        console.log('No data!');
      }
      that.get_distribution((data) => {
        DataLoader.fetch_label(() => {
          DataLoader.draw_batch_density(DataLoader.curData, 1000, 500)
          DataLoader.refresh_scatter_color()
        });
      });
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idxes': Array.from(BrushController.selected_idx),
    });
    node.notify();
  };

  that.adapt = function () {
    const node = new request_node(that.adapt_url, (data) => {
      console.log('adapt');
      d3.select('#show_comparsion').attr('disabled', null);

      let localCallback = undefined;
      if (that.PerfStatus) {
        that.PerfStatus.old_data = null;
        that.PerfStatus.new_data = null;
        localCallback = that.update_squares_view;
      }

      if (DataLoader.previous_win_size != DataLoader.win_size) {
        DataLoader.change_win_size();
        if (localCallback) localCallback();
      } else {
        that.update_table(function() {
          DataLoader.update_line_plot(localCallback);
        });
        //DataLoader.update_line_plot();
      }
      //if (that.PerfStatus.old_data) {
      //  that.update_squares_view("new_data")
      //} else {
      //  that.clear_square_view()
      //}
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idxes': BrushController.selected_idx,
    });
    node.notify();
  };

  that._createModelTableTD = function (rows) {
    rows.selectAll('td')
      .data((d, i) => [i, d['count']]) // , that.generate_btn(d, i)])
      .enter()
      .append('td')
      .style('text-align', 'center')
      .style('vertical-align', 'middle')
      .html(function (d) {
        return d;
      });

    rows.insert('td', ':nth-child(2)')
      .attr('align', 'center')
      .append('div')
      .style('width', '25px')
      .style('height', '25px')
      .style('margin', 'auto')
      .append('svg')
      .style('width', '25px')
      .style('height', '25px')
      .append('rect')
      .attr('width', '25px')
      .attr('height', '25px')
      .style('fill', (d, i) => that.model_colors[i]);
    rows.append('td').attr('align', 'center').each(function (d) {
      that.draw_percentage(d3.select(this), d['gaussian_percentage'], Array.from(new Set(that.gmm_label)).sort((a, b) => a - b).map(x => DataLoader.color(x)));
    });

    // for combox box
    const cbxTD = rows
      .append('td')
      .append('div')
      .style('text-align', 'center');
    cbxTD.append('input')
      .attr('type', 'checkbox')
      .attr('class', 'filled-in checkbox-blue-grey')
      .property('checked', (d) => (d.use))
      .on('click', function (d) {
        //const input = d3.select(this).select('input');
        //const flag = input.property('checked');
        //input.property('checked', !flag);
        // d.use = !d.use;
        that.toggle_model(d.id, d3.select(this).property('checked'));
      });
    cbxTD.append('span').style('vertical-align', 'middle');
  };

  that.draw_percentage = function (selector, percentage, colors) {
    if (colors) {
      const hex2rgba = (hex, alpha) => {
        const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
        return `rgba(${r},${g},${b},${alpha})`;
      };
      colors = colors.map(color => hex2rgba(color, 0.5));
    } else {
      colors = that.model_colors;
      let checkboxes = d3.select('#model-tbody').selectAll('input')._groups[0];
      let new_percentage = [];
      let j = 0;
      for (let i = 0; i < checkboxes.length; ++i) {
        if (checkboxes[i].checked) {
          new_percentage.push(percentage[j]);
          ++j;
        } else {
          new_percentage.push(0);
        }
      }
      percentage = new_percentage;
    }
    const width = 150;
    const widthes = percentage.map(x => x * width);
    const cumulativeSum = (sum => value => sum += value)(0);
    const endX = widthes.map(cumulativeSum);

    let rect = selector
      .append('div')
      .style('width', width + 'px')
      .style('height', '25px')
      .style('margin', 'auto')
      .append('svg')
      .style('width', width + 'px')
      .style('height', '25px')
      .selectAll('rect')
      .data(percentage)
      .enter()
      .append('rect')
      .attr('x', function(d, i) {
        return  endX[i] - widthes[i];
      })
      .attr('width', function(d, i) {
        return widthes[i]
      })
      .attr('height', 25)
      .style('fill', function(d, i) {
        return  colors[i];
      });
  }

  that.add_model = function () {
    $("#loading")[0].style.display = "block";
    const node = new request_node(that.add_model_url, (data) => {
      $("#loading")[0].style.display = "none";
      console.log('add model');
      if (data) {
        const model = data['model'];
        that.models.push(model);
        that.set_model(that.models);
      } else {
        console.log('No data!');
      }
      //window.ActionTrail.notify('new-model', that.models.length - 1);
      // Weikai: actiontrail.notify will clean the focus for line chart.
      // I will preserve it if needed
      //BrushController.update_line_brush(null);
      //that.clear_square_view();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idx': Array.from(BrushController.selected_idx),
    });
    node.notify();
  };

  that.update_table = function (callback) {
    const node = new request_node(that.update_table_url, (data) => {
      console.log('update_table');
      if (data) {
        that.set_chunks(data['chunks']);
        that.set_model(data['models']);
      } else {
        console.log('No data!');
      }
      if (callback) callback();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  };


  that.update_single_chunk_info = function (idx, callback) {
    if (!idx || idx.length == 0) return;
    const node = new request_node(that.single_chunk_info_url, (data) => {
      console.log('update_single_chunk_info');
      if (data) {
        that.chunks[0] = data;
        that.set_chunks(that.chunks);
      } else {
        console.log('No data!');
      }
      if (callback) callback();
    }, 'json', 'POST');
    node.set_data({
      'idx': idx
    });
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  };

  that.update_model_list = function () {
    const max_t = Math.max(...DataLoader.curData.map((x) => x.t));
    for (const chunk of that.chunks) {
      const chunk_max_t = Math.max(...chunk.idx);
      chunk['valid'] = chunk_max_t <= max_t;
    }
    d3.select('#drift-tbody')
      .selectAll('tr.content')
      // .style('pointer-events', d => d['valid'] ? null : 'none')
      .style('background-color', (d) => d['valid'] ? null : 'lightgrey')
      .style('cursor', (d) => d['valid'] ? null : 'not-allowed');
  };

  that.get_distribution = function (callback) {
    const node = new request_node(that.get_distributions_url, (data) => {
      console.log('get_distribution');
      if (data) {
        that.set_distributions(data['distributions']);
      } else {
        console.log('No data!');
      }
      if (callback) callback();
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  };

  that.get_predict_vector = function (end_idx) {
    const node = new request_node(that.get_predict_vector_url, (data) => {
      console.log('get predict vector');
      if (data) {
        console.log(data);
      } else {
        console.log('No data!');
      }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'end_idx': end_idx,
    });
    node.notify();
  };

  that.get_predict_stat = function () {
    const node = new request_node(that.get_predict_stat_url, (data) => {
      console.log('get predict stat');
      if (data) {
        console.log(data);
      } else {
        console.log('No data!');
      }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idx': BrushController.selected_idx,
    });
    node.notify();
  };

  //function getRandomArbitrary(min, max) {
  //  return Math.random() * (max - min) + min;
  //}
  //
  //that.get_squares_data = function(idx) {
  //  const use_fake_data = true;
  //  if (use_fake_data) {
  //    const n_bin = 10
  //    const n_classes = 5;
  //    let classes = []
  //    for (let i = 0; i < n_classes; ++i) {
  //      classes.push('' + i)
  //    }
  //    let squares = {}
  //    classes.forEach((cl) => {
  //      let size = Math.floor(idx.length / n_classes)
  //      let k = getRandomArbitrary(.6, .9)
  //      let left = size
  //      let cldata = {}
  //      for (let i = 0; i < n_bin; ++i) {
  //        let cur_size = Math.floor(left * k)
  //        left -= cur_size
  //        cldata['bin' + i] = { TP: cur_size }
  //        if (i + 2 == n_bin) {
  //          k = 1;
  //        } else {
  //          k = getRandomArbitrary(.3, .6)
  //        }
  //      }
  //      for (let i = 0; i < n_bin; ++i) {
  //        let FP = [], FN = []
  //        for (let j = 0; j < n_classes; ++j) {
  //          FP.push(Math.floor(getRandomArbitrary(0, 0.05) * cldata['bin' + i].TP))
  //          FN.push(Math.floor(getRandomArbitrary(0, 0.05) * cldata['bin' + i].TP))
  //        }
  //        cldata['bin' + i].FP = FP
  //        cldata['bin' + i].FN = FN
  //      }
  //      squares[cl] = cldata
  //    })
  //    return { classes, squares }
  //  }
  //};
  that.get_square_data = function (callback) {
    const node = new request_node(that.get_square_data_url, (data) => {
      console.log('get square data');
      if (data) {
        if ('current_result' in data) {
          that.PerfStatus['new_data'] = data['current_result'];
        }
        if ('previous_result' in data) {
          that.PerfStatus['old_data'] = data['previous_result'];
        }
        callback();
      } else {
        console.log('No data!');
      }
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'idxes': that.PerfStatus.idx,
    });
    node.notify();
  }

  that.toggle_show_comparsion = function () {
    if (!that.PerfStatus) return;
    that.performance_comparsion = !that.performance_comparsion;
    if (that.PerfStatus.old_data === null) that.performance_comparsion = false;
    d3.select('#show_comparsion').html(that.performance_comparsion ? "Hide Comparsion" : "Show Comparsion");
    that._update_squares_view()
  }
  d3.select('#show_comparsion').on('click', that.toggle_show_comparsion);

  that.update_squares_view = function () {
    if (!that.PerfStatus) {
      //d3.select("#performance-view .view-body").selectAll('*').remove()
      return;
    }
    if (that.PerfStatus.old_data === null && that.PerfStatus.new_data === null) {
      that.get_square_data(that._update_squares_view);
    } else {
      that._update_squares_view();
    }
  }

  that.clear_square_view = function () {
    that.PerfStatus = null;
    that._update_squares_view();
  }

  that._update_squares_view = function () {
    const view_width = $("#performance-view .view-body").width() - 2
    const view_height = $("#performance-view .view-body").height() - 10
    let margin = {
        top: 10,
        right: 20,
        bottom: 30,
        left: 45
      },
      width = view_width - margin.left - margin.right,
      height = view_height - margin.top - margin.bottom;

    d3.select("#performance-view .view-desc").select("*").remove()
    d3.select("#performance-view .view-desc").append("p")
      .style("padding-left", "20px")
      .style("font-size", "14px")
      .text(() => {
        if (!that.PerfStatus) {
          return "Click a sample to show its performance."
        } else if (that.performance_comparsion) {
          return `Current Model Acc: ${Math.round(ModelLoader.PerfStatus['new_data']['acc']*100)/100}.  Previous Model Acc: ${Math.round(ModelLoader.PerfStatus['old_data']['acc']*100)/100}`
        } else {
          return `Current Model Acc: ${Math.round(ModelLoader.PerfStatus['new_data']['acc']*100)/100}`
        }
      })

    d3.select("#performance-view .view-body").select("*").remove()
    let svg = d3.select("#performance-view .view-body").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    if (!that.PerfStatus || !that.PerfStatus.new_data) {
      return
    }

    const old_data = that.PerfStatus.old_data
    const new_data = that.PerfStatus.new_data
    const classes = new_data.classes
    const n_class = classes.length
    //const bar_padx = width / n_class * 0.4
    //const bar_widthx = width / n_class - bar_padx
    const bar_block = width * 0.9 / n_class
    const bar_widthx = width * 0.9 / n_class / 2 - 10;
    const bar_padx = width * 0.1 + bar_widthx;

    function get_aggregated_data(data) {
      return [].concat(...Object.keys(data.squares).map((cl, clno) =>
        Object.keys(data.squares[cl]).map(bin => ({
          class: cl,
          clno: clno,
          bin: bin,
          TP: data.squares[cl][bin].TP,
          FP: data.squares[cl][bin].FP,
          FN: data.squares[cl][bin].FN,
          TP_idx: data.squares[cl][bin].TP_idx,
          FP_idx: [].concat(...data.squares[cl][bin].FP_idx),
          FN_idx: [].concat(...data.squares[cl][bin].FN_idx),
          FP_sum: data.squares[cl][bin].FP.reduce((a, b) => a + b, 0),
          FN_sum: data.squares[cl][bin].FN.reduce((a, b) => a + b, 0),
        }))
      ))
    }

    let agg_new_data = get_aggregated_data(new_data)
    let agg_old_data
    const bins = [...new Set(agg_new_data.map(d => d.bin))].sort((a, b) => a > b ? -1 : 1)
    //let max_bin = Math.max(...agg_new_data.map(d => d.TP + d.FP_sum))
    const _get_max_bin = (agg_data) => (Math.max(
      Math.max(...agg_data.map(d => d.TP + d.FP_sum)),
      Math.max(...agg_data.map(d => d.FN_sum)),
      10)); // 10 is used for avoiding too less error.
    let max_bin = _get_max_bin(agg_new_data);

    if (that.performance_comparsion) {
      agg_old_data = get_aggregated_data(old_data)
      max_bin = Math.max(max_bin,
        _get_max_bin(agg_old_data)
      )
    }

    if (max_bin>40) {
      max_bin=40
      for(let d of agg_new_data) {
        d.FN_sum = Math.min(d.FN_sum, max_bin);
        d.FP_sum = Math.min(d.FP_sum, max_bin);
        d.TP = Math.min(d.TP, max_bin);
      }
      if (agg_old_data)
      for(let d of agg_old_data) {
        d.FN_sum = Math.min(d.FN_sum, max_bin);
        d.FP_sum = Math.min(d.FP_sum, max_bin);
        d.TP = Math.min(d.TP, max_bin);
      }
    }


    function draw(data, view, height, width, drawlabel = true) {
      let x = d3.scaleLinear()
        .range([0, bar_widthx])
        .domain([0, max_bin])

      let y = d3.scaleBand()
        .range([0, height])
        .padding(0.1)
        .domain(bins)

      let color = '30,30,30'

      let class_labels = view.selectAll(".label")
        .data(classes)
        .enter().append("g").attr("class", "label")

      class_labels.append("line")
        .attr("x1", (d, i) => bar_block * i + bar_padx)
        .attr("y1", 0)
        .attr("x2", (d, i) => bar_block * i + bar_padx)
        .attr("y2", height)
        .style("stroke", `rgb(${color})`)
        .style("stroke-width", 1)

      if (drawlabel){
        class_labels.append("text")
        .attr("x", (d, i) => bar_block * i + bar_padx)
        .attr("y", height)
        //.attr("dx", d => -d.length * 3)
        .attr("dy", 15)
        .style('text-anchor', 'middle')
        .style("font-weight", 400)
        .style("font-size", 14)
        .style("fill", `rgb(${color})`)
        .text(d => d)
      const perf_text = 'confidence'
      view.append("text")
        .attr("x", 10)
        .attr("y", height)
        //.attr("dx", -perf_text.length * 3)
        .attr("dy", 15)
        .style('text-anchor', 'middle')
        .style("font-size", 14)
        .style("font-weight", 400)
        .style("fill", `rgb(${color})`)
        .text(perf_text)
      }
      //if (drawlabel) {
      //  view.append("text")
      //  .attr("x", -100)
      //  .attr("y", -20)
      //  .style("font-weight", 400)
      //  .style("font-size", 16)
      //  .style("fill", `rgb(${color})`)
      //  .attr('transform', 'rotate(270)')
      //  .text('Current')
      //} else {
      //  view.append("text")
      //  .attr("x", -0)
      //  .attr("y", -20)
      //  .style("font-weight", 400)
      //  .style("font-size", 16)
      //  .style("fill", `rgb(${color})`)
      //  .attr('transform', 'rotate(-270)')
      //  .text('Previous')
      //}



      // append the rectangles for the bar chart
      const bargroup = view.selectAll(".bargroup")
        .data(data)
        .enter().append("g")
        .attr("class", "bargroup")

      function onmouseover_bar(idx) {
        BrushController.refresh_scatter(idx)
        //that.get_predict_vector(idx, draw_paraller)
      }

      function draw_paraller(para_data) {
          return;
        view.selectAll('.parallel')
          .data(para_data).enter()
          .append('path')
          .attr('class', 'parallel')
          .attr('d', (d) =>
            'M' + d.slice(0, n_class).map((e, i) => `${bar_block * i + bar_padx} ${y('bin' + e)}`).join('L') + 'Z'
          )
          .attr('fill', 'none')
          .style('stroke', `rgba(${color}, 0.5)`)
          .style('stroke-width', 0.3)
      }

      function onmouseout_bar() {
        BrushController.reset_brush()
        view.selectAll('.parallel').remove()
      }

      bargroup.append("rect")
        .attr("x", d => bar_block * d.clno + bar_padx)
        .attr("y", d => y(d.bin))
        .attr("width", d => x(d.FP_sum))
        .attr("height", y.bandwidth())
        .attr("fill", `rgba(${color},0.2)`)
        .style("stroke", `rgb(${color})`)
        .style("stroke-width", .3)
        .on("mouseover", function (d) {
          d3.select(this).style("fill", `rgba(${color},0.4)`)
          onmouseover_bar(d.FP_idx);
        })
        .on("mouseout", function (d) {
          d3.select(this).style("fill", `rgba(${color},0.2)`)
          onmouseout_bar();
        })

      bargroup.append("rect")
        .attr("x", d => bar_block * d.clno + bar_padx + x(d.FP_sum))
        .attr("y", d => y(d.bin))
        .attr("width", d => x(d.TP))
        .attr("height", y.bandwidth())
        .attr("fill", `rgba(${color},0.4)`)
        .style("stroke", `rgb(${color})`)
        .style("stroke-width", .3)
        .on("mouseover", function (d) {
          d3.select(this).style("fill", `rgba(${color},0.6)`)
          onmouseover_bar(d.TP_idx);
        })
        .on("mouseout", function (d) {
          d3.select(this).style("fill", `rgba(${color},0.4)`)
          onmouseout_bar();
        })

      bargroup.filter(d => x(d.FN_sum) > 1)
        .append("rect")
        .attr("x", d => bar_block * d.clno + bar_padx - x(d.FN_sum))
        .attr("y", d => y(d.bin))
        .attr("width", d => Math.max(1, x(d.FN_sum)))
        .attr("height", y.bandwidth())
        .style("fill", `rgba(${color},0)`)
        .style("stroke", `rgb(${color})`)
        .style("stroke-width", .3)
        .on("mouseover", function (d) {
          d3.select(this).style("fill", `rgba(${color},0.2)`)
          onmouseover_bar(d.FN_idx);
        })
        .on("mouseout", function (d) {
          d3.select(this).style("fill", `rgba(${color},0)`)
          onmouseout_bar();
        })

      view.transition().duration(500)
        .style("opacity", 1)

      // add the y Axis
      view.append("g")
        .call(d3.axisLeft(y).tickFormat(d => d.slice(3)));
    }

    if (that.performance_comparsion) {
      //let new_max_bin = Math.max(...agg_new_data.map(d => d.TP + d.FP_sum))
      // let old_max_bin = _get_max_bin(agg_old_data);
      // max_bin = Math.max(max_bin, old_max_bin);
      let paddingForText = 17;
      let new_view = svg
        .append("g")
        .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")")
        .style("opacity", 0)
      let old_view = svg
        .append("g")
        .attr("transform",
          "translate(" + margin.left + "," + (margin.top + height / 2 + paddingForText) + ")")
        .style("opacity", 0)
      draw(agg_old_data, old_view, (height - 10) / 2, width, false)
      draw(agg_new_data, new_view, (height - 10) / 2, width)
    } else {
      let new_view = svg
        .append("g")
        .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")")
        .style("opacity", 0)
      draw(agg_new_data, new_view, height, width)
    }

    console.log('------------- kelei test: ',
      new_data.acc.toFixed(4),
      that._get_current_accuracy(new_data).toFixed(4),
      (that.performance_comparsion?that._get_current_accuracy(old_data).toFixed(4):'None'));
  };

  that._get_current_accuracy = function (data) {
    let count = 0,
      tCount = 0;
    for (let cls of data.classes) {
      for (let bin in data.squares[cls]) {
        count += data.squares[cls][bin].FP.reduce((a, b) => (a + b), 0);
        tCount += data.squares[cls][bin].TP;
      }
    }
    return (tCount / (count + tCount));
  };

  that.set_distributions = function (data) {
    that.distributions = data;
  };

  that.onmouseover = (d) => {
    BrushController.refresh_scatter(d.idx);
  };
  that.onmouseout = () => {
    BrushController.refresh_scatter();
  };

  that.render_border = (d) => {
    const red = '2px solid rgba(204,98,87, 0.8)';
    const blue = '2px solid rgba(87,135,204, 0.8)'
    let nodes = d3.select('#drift-tbody').selectAll('tr.content').nodes()
    if (DataLoader.current_view == 'density diff') {
      for (let i = 0; i < nodes.length; ++i) {
        $(nodes[i])[0].style.border = 'none'
      }
      $(nodes[nodes.length - 1])[0].style.border = 'red'
      if (d && d.id != nodes.length) {
        that.compared_batch = d.id - 1
        that.compared_batch_idx = d.idx
      } else {
        that.compared_batch = nodes.length - 2
        that.compared_batch_idx = null
      }
      DataLoader.draw_density_grid()
      $(nodes[that.compared_batch])[0].style.border = blue
    } else {
      for (let i = 0; i < nodes.length; ++i) {
        $(nodes[i])[0].style.border = 'none'
      }
      if (d) {
        $(nodes[d.id])[0].style.border = 'red'
      }
    }
  };
  that.onclick = (d) => {
    if (DataLoader.current_view != 'density diff') {
      that.PerfStatus = {
        id: d.id,
        idx: d.idx,
        old_data: null,
        new_data: null
      }
      that.data_visibility = that.data_visibility || {}
      that.data_visibility[d.id] = that.data_visibility[d.id] || false
      //that.update_squares_view('old_data');
      that.update_squares_view()
      if (!that.data_visibility[d.id]) {
        BrushController.scatter_show(d.idx)
        that.data_visibility[d.id] = true
      } else {
        BrushController.scatter_hide(d.idx)
        that.data_visibility[d.id] = false
      }
      BrushController.refresh_scatter();
    } else {

    }
    that.render_border(d)
  };
};