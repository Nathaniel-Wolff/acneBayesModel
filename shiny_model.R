library(shiny)
library(stevedore)
library(httr2)
library(ggplot2)
library(jsonlite)

system("docker rm -f acne_bayes_server > /dev/null 2>&1")

# 2. Spin up the container detached
message("🚀 Spinning up container via OrbStack...")
system("docker run -d --name acne_bayes_server -p 127.0.0.1:8000:8000 acne-model:latest")

# 3. Dynamic polling loop with an extended training window
message("⏳ Waiting for HBM training layer to converge (JAX/Optimization)...")
server_ready <- FALSE
attempts <- 0
max_attempts <- 90 # Generous 45-second ceiling for full convergence step

while(!server_ready && attempts < max_attempts) {
  attempts <- attempts + 1
  
  tryCatch({
    # Explicitly use 127.0.0.1 instead of localhost to prevent ipv6 loop issues
    res <- request("http://127.0.0.1:8000/docs") %>% 
      req_timeout(0.4) %>% 
      req_perform()
    
    if (res$status_code == 200) {
      server_ready <- TRUE
    }
  }, error = function(e) {
    # Print a tiny visual pulse every 3 seconds so you know R hasn't hung
    if (attempts %% 6 == 0) {
      message("   ...still optimizing model parameters...")
    }
    Sys.sleep(0.5)
  })
}

if (server_ready) {
  message("✅ Acne Bayes Server is fully trained and responsive after ", (attempts * 0.5), " seconds!")
} else {
  # CRITICAL DIAGNOSTIC: If it fails, print out what Python is currently screaming
  message("❌ Server initialization timed out! Extracting live container logs:")
  system("docker logs acne_bayes_server")
  stop("App halted: Container failed to become ready in time.")
}

# 4. Global teardown hook to clean up when the user exits the app
onStop(function() {
  message("🛑 Stopping and removing OrbStack container safely...")
  system("docker stop acne_bayes_server > /dev/null 2>&1")
  system("docker rm acne_bayes_server > /dev/null 2>&1")
  message("✨ Cleanup complete!")
})

ui <- fluidPage(titlePanel("Acne Bayes Model - Server Logic Test"), 
                sidebarLayout(
                  sidebarPanel(
                    sliderInput("latent_state_0", "Initial Latent State Bacterial State (B_t) Baseline", min = -5.0, max = 5.0, value = 2.0, step = 0.1),
                    sliderInput("latent_state_1", "Initial Latent State Sebum State (S_t) Baseline", min = -5.0, max = 5.0, value = 2.0, step = 0.1),
                    sliderInput("latent_state_2", "Initial Latent State Inflammatory State (I_t) Baseline", min = -5.0, max = 5.0, value = 2.0, step = 0.1),
                    
                    sliderInput("isotret_dosage", "Isotretinoin Dosage (mg)", min = 0.001, max = 1.0, value = 0.05, step = 0.01),
                    sliderInput("clin_bpo_dosage", "Clindamycin-Benzoyl Peroxide Dosage (mg)", min = 0.001, max = 1.0, value = 0.05, step = 0.01),
                    sliderInput("days", "Regimen Days", min = 1, max = 14, value = 2, step = 1)
                  ), 
      
                            mainPanel(plotOutput("severity_plot")
              ) 
            )
          )


#Shiny client and reactivity 

model_server <- function(input, output, session) {
  python_output_conditions <- reactive( {
    #creating dummy treatment matrix
    isotret_row <- c(rep(input$isotret_dosage, input$days))
    clin_bpo_row <- c(rep(input$clin_bpo_dosage, input$days))
    
    dummy_treatment_matrix <- cbind(isotret_row, clin_bpo_row)
    
    treatment_ls_payload <- list(
      initial_latent_state = c(input$latent_state_0, input$latent_state_1, input$latent_state_2), 
      treatment_series = dummy_treatment_matrix
    )
    
    #POST request
    req <- request("http://localhost:8000/send_predictions") %>%
      req_body_json(treatment_ls_payload)
    
    response <- req_perform(req)
    raw_json_CIs <- resp_body_json(response, simplifyVector = TRUE)
    usable_ci_data <- raw_json_CIs$credible_intervals
  
  #parse response into dataframe for plotting
    df <- data.frame(
    time = as.numeric(names(usable_ci_data)),
    expected_severity = sapply(usable_ci_data, function(x) x[1]),
    lower_ci = sapply(usable_ci_data, function(x) x[2]),
    upper_ci = sapply(usable_ci_data, function(x) x[3]))
  return(df)})

#total run
  output$severity_plot <- renderPlot({
    df <- python_output_conditions()
    
    # Enhanced ggplot to visualize both expected mean and the model's 95% Credible Intervals
    ggplot(df, aes(x = time, y = expected_severity)) +
      geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "#3498db", alpha = 0.2) +
      geom_line(color = "#2c3e50", linewidth = 1.2) +
      theme_minimal() +
      labs(
        title = "Modeled Acne Severity Over Time",
        x = "Days", 
        y = "Latent State Severity Projection",
        caption = "Shaded region represents 95% Credible Interval"
      )
  })
}

#run 
shinyApp(ui, model_server)




