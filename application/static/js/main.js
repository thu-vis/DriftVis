/*
 * added by changjian, 20190520
 * */

function set_up() {
  //const datasets = ['synthetic', 'electricity', 'weather', 'SVD'];
  //const datasets = ['weather', 'paper'];
  const datasets = ['weather'];
  const methods = ['Value', 'Drift'];
  // const attr_methods = ['CUSUM', 'TVD'];
  // const non_attr_methods = ['GMM'];
  // const methods = attr_methods.concat(non_attr_methods);

  DataLoader = new DataLoaderClass();
  ModelLoader = new ModelLoaderClass();
  BrushController = new BrushControllerClass();
  $('#filter-select-dataset')
    .html(datasets.map((x) => '<option>' + x + '</option>').join(''))
    .selectpicker('refresh');
  $('#filter-select-dataset').on('change', function (e) {
    DataLoader.set_dataset($('#filter-select-dataset').val());
  });
  $('#filter-select-method')
    .html(methods.map((x) => '<option>' + x + '</option>').join(''))
    .selectpicker('refresh');
  $('#filter-select-method').on('change', function (e) {
    let val = $('#filter-select-method').val();
    DataLoader.toggle_candidate_key(methods.indexOf(val));
    DataLoader.update_line_plot();
  });
  ModelLoader.set_introduction(d3.select('#detail-introduction'));

  // let names = [
  //    '#filter-select-attribute',
  //    '#filter-select-driftdegree'];
  // $('#filter-select-method').on('change', function (e) {
  //    let old_attr = [ $(names[0]).val(), $(names[1]).val() ];
  //    let selected_method = $('#filter-select-method').val();
  //    let html_string = [
  //        `<optgroup label="Origin">${DataLoader.origin_attributes.map(x => '<option>' + x + '_Origin</option>').join('')}</optgroup>`,
  //        ''];
  //    for (let method of selected_method) {
  //        if (attr_methods.includes(method)) {
  //            html_string[1] += `<optgroup label="${method}">${DataLoader.origin_attributes.map(x => '<option>' + x + "_" + method + '</option>').join('')}</optgroup>`;
  //        } else {
  //            html_string[1] = `<option>${method}</option>` + html_string[1];
  //        }
  //    }
  //    names.forEach((name, i)=>{
  //        $(name).html(html_string[i]).selectpicker('refresh');
  //        if (old_attr[i].length > 0) {
  //            $(name).selectpicker('val', old_attr[i]).trigger('change');
  //        }
  //    });
  // });
  $('#add-model').on('click', ModelLoader.add_model);
  // names.forEach((name)=>{
  //    $(name).on('change', function (e) {
  //        DataLoader.update_line_plot();
  //    });
  // });
  /*
  $('#play_or_pause').html('Next step');
  $('#play_or_pause').on('click', function (e) {
    
    $('#next').attr('disabled', false);
    $('#play_or_pause').attr('disabled', true);
    //console.log(window.play_or_pause_status)
    //if (!window.play_or_pause_status) {
    //  window.play_or_pause_status = 1
    //  $('#play_or_pause').html("Play")
    //} else {
    //  window.play_or_pause_status = 0
    //  $('#play_or_pause').html("Pause")
    //}
  });
  */
  $('#next').on('click', function (e) {
    DataLoader.next_data(()=>{
      //$('#next').attr('disabled', true);
      $('#play_or_pause').attr('disabled', false);
     });
    //ModelLoader.clear_square_view()
  });
  $('#merge').on('click', function (e) {
    ModelLoader.adjust_component();
    //ModelLoader.clear_square_view()
  });
  $('#adapt').on('click', function (e) {
    ModelLoader.adapt();
  });

  $('#popover_init_ok').on('click', function(e){
    $('#popover_init')[0].style.display = 'none'
  })

  /*
  $('#show_density_diff').on('click', function(e) {
    if (window.show_density_diff) {
      $('#show_density_diff').html("Show Density Diff")
      window.show_density_diff = false;
      $(".density-plot")[0].style.display = "block"
      $(".plot")[0].style.display = "block"
      $(".density-hazes")[0].style.display = "block"
      $(".density-heatmap")[0].style.display = "none"
    } else {
      $('#show_density_diff').html("Hide Density Diff")
      window.show_density_diff = true;
      $(".density-plot")[0].style.display = "none"
      $(".plot")[0].style.display = "none"
      $(".density-hazes")[0].style.display = "none"
      $(".density-heatmap")[0].style.display = "block"
    }
  })
  $("#show_density_diff").attr("disabled", true)
  */
  //$('#model-scroll-div').height($('#model-scroll-div').height());
  //$('#detail-scroll-div').height($('#detail-scroll-div').height());
  DataLoader.set_dataset(datasets[0]);
  $('#filter-select-dataset').selectpicker('val', datasets[0]);
  $('#filter-select-method').selectpicker('val', methods[1]);

  function setComponentSize(){
    let components = [...$(".component")]
    let comp_parts = [...$(".component-part")]
    for (let c of components) {
      const r = c.getBoundingClientRect();
      c.style['max-width'] = `${r.width}px`;
      c.style['max-height'] = `${r.height}px`;
    }
    for (let c of comp_parts) {
      const r = c.getBoundingClientRect();
      c.style['max-width'] = `${r.width}px`;
      c.style['max-height'] = `${r.height}px`;
    }
  }
  $(window).resize(setComponentSize);
  setTimeout(setComponentSize, 10);
}

// main (entry of the application)
$(document).ready(function () {
  set_up();
});